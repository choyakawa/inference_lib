// move reduction back into a separate kernel.
// in terms of pure kernel times, this is faster
#include "common.h"
#include "vec.cuh"
#include "cuda_check.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline_primitives.h>

namespace cg = cooperative_groups;

namespace v21
{
constexpr const int SubWarpSize = 8;
constexpr const int WarpSize = 32;

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void hogwild_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const int* fragment_lengths,
        const scalar_t* const* key_fragments,
        const scalar_t* const* value_fragments,
        Shape shape) {
    // Input:   keys: [Hkv, fragment_lengths[i], E] for i in [F]
    //          values: [Hkv, fragment_lengths[i], Ev] for i in [F]
    //          fragment_lengths: [F]
    //          queries: [F, W, Hq, S, E]
    //          locations [F, W, S]
    // Scratch: workspace [W, Hq, S, Ev] (in float32, iff scalar_t != float32) + [W, Hq, S] BlockResult
    // Output:  [W, Hq, S, Ev]
    // attention mask: s attends to l iff locations[b, s] >= l (i.e., shifted causal masking)

    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S;
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    ptrdiff_t q_stride = E * S * Hq * W;
    extern __shared__ float scratch[];

    // adjust scale so we can use base-2 exponent later on
    float l2scale = scale / std::log(2.f);

    int hkv = blockIdx.x;
    int w = blockIdx.y % W;
    int s = blockIdx.y / W;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int hq = hkv * GQA;
    ptrdiff_t q_offset = ((w * Hq + hq) * S + s) * E;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);   // vectors per head per thread
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);  // vectors per head per thread

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using full_fvec_t = GenericVector<float, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>;
    qk_cache_t q_cache[GQA];

    // combine values
    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    // determine maximum and online logsumexp
    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

        for (int f = 0; f < shape.F; ++f) {
            int q_loc = locations[(f * W + w) * S + s];
            int L = fragment_lengths[f];
            int maxL = std::min(L, q_loc + 1);

            for (int gqa = 0; gqa < GQA; ++gqa) {
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    full_vec_t qv = full_vec_t::load(queries + f * q_stride + q_offset + gqa * S * E + e);
                    for (int j = 0; j < VecSize; ++j) {
                        q_cache[gqa][ee * VecSize + j] = qv[j];
                    }
                }
            }

            const scalar_t* value_fragment = value_fragments[f];
            const scalar_t* key_fragment = key_fragments[f];

            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;
                ptrdiff_t k_offset = (hkv * L + l) * E;
                ptrdiff_t v_offset = (hkv * L + l) * Ev;
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x,
                                            key_fragment + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x,
                                            value_fragment + v_offset + e, sizeof(full_vec_t));
                }
            };

            int stage = 0;
            ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
            __pipeline_commit();
            ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
            __pipeline_commit();

            for (int ll = split; ll < maxL; ll += StepSize) {
                int l = ll + sub_warp.meta_group_rank() * splits;
                qk_cache_t keys;
                v_cache_t vals;
                __pipeline_wait_prior(1);
                if (l >= maxL) continue;
                unsigned mask = __activemask();

                for (int ee = 0; ee < VPH_k; ++ee) {
                    full_vec_t tmp = keys_lookahead[(stage * VPH_k + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) {
                        keys[ee * VecSize + j] = (float)tmp[j];
                    }
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    full_vec_t tmp = vals_lookahead[(stage * VPH_v + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) {
                        vals[ee * VecSize + j] = (float)tmp[j];
                    }
                }

                ldg_sts((stage + 2) % 2, l + 2 * StepSize);
                stage = (stage + 1) % 2;
                __pipeline_commit();

                float qk[GQA] = {};
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    for (int ee = 0; ee < VPH_k; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            qk[gqa] += q_cache[gqa][ee * VecSize + j] * keys[ee * VecSize + j];
                        }
                    }
                }

                // important: By having the warp shuffles together like this in a separate loop,
                // the compiler ends up generating better sequenced assembly, where we first initiate a
                // bunch of shuffles and only then do the addition, hiding the latency much better
                // than in the single-loop version.
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0100, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0010, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0001, 8);
                }

                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    if (qk[gqa] > maximum[gqa]) {
                        float rescale = std::exp2f(l2scale * (maximum[gqa] - qk[gqa]));
                        for (int j = 0; j < v_cache_t::size; ++j) {
                            v_cache[gqa][j] *= rescale;
                        }
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));

                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                        }
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        // combine split-k results
        if (sub_warp.thread_rank() == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + 2 * sub_warp.meta_group_rank() + 2 * WarpSize * gqa);
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        float r_max = maximum[gqa];
        float l_max = maximum[gqa];
        float r_lse = 0;
        if (warp.thread_rank() < SubWarpMetaSize) {
            stats_t data = stats_t::load(scratch + 2 * warp.thread_rank() + 2 * WarpSize * gqa);
            r_max = data[0];
            r_lse = data[1];
        }

        maximum[gqa] = cg::reduce(warp, r_max, cg::greater<float>{});
        r_lse *= std::exp2f(l2scale * (r_max - maximum[gqa]));
        lse[gqa] = cg::reduce(warp, r_lse, cg::plus<float>{});

        // Note: It *is* possible that no thread in this warp had any valid position (due to causal masking),
        // which would lead to division by zero -> 0 * inf = NaN here.
        if (lse[gqa] != 0) {
            float rescale = std::exp2f(l2scale * (l_max - maximum[gqa])) / lse[gqa];
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[gqa][j] *= rescale;
            }
        }

        if (threadIdx.x == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        }

        // now reduce value across subwarp within a warp
        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_cache[gqa][ee * VecSize + j];
                static_assert(SubWarpSize == 8);
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_cache[gqa][ee * VecSize + j] = v;
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.meta_group_rank() % (WarpSize / SubWarpSize) == 0) {
            for (int ee = 0; ee < VPH_v; ++ee) {
                int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                full_fvec_t store;
                for (int j = 0; j < VecSize; ++j) {
                    store[j] = v_cache[gqa][ee * VecSize + j];
                }
                store.store(scratch + e + Ev * sub_warp.meta_group_rank() / (WarpSize / SubWarpSize) + gqa * 256 / WarpSize * Ev);
            }
        }
    }
    __syncthreads();

    // Let all warps in the block collaborate to reduce all GQA groups.
    // Each warp (ID 0-7 for a 256-thread block) will handle a different GQA group in each iteration of the loop.
    for (int gqa_offset = 0; gqa_offset < GQA; gqa_offset += (blockDim.x / WarpSize)) {
        int gqa = warp.meta_group_rank() + gqa_offset;
        if (gqa >= GQA) {
            continue;
        }

        int h = hkv * GQA + gqa;
        int res_base = ((w * Hq + h) * S + s);
        int res_inc = W * Hq * S;
        int res_idx = res_base + split * res_inc;
        float* global_accumulator = reinterpret_cast<float*>(workspace);
        float* lse_target = global_accumulator + W * Hq * S * Ev * splits;

        stats_t data = stats_t::load(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        float own_lse = data[1];
        float own_max = data[0];
        own_lse = std::log2(own_lse) + l2scale * own_max;

        for (int e = vec_t::size * warp.thread_rank(); e < Ev; e += vec_t::size * warp.size()) {
            // merge the local results
            fvec_t res = fvec_t::zeros();
            for (int j = 0; j < SubWarpMetaSize / (WarpSize / SubWarpSize); ++j) {
                fvec_t sv = fvec_t::load(scratch + e + Ev * j + gqa * 256 / WarpSize * Ev);
                for (int jj = 0; jj < vec_t::size; ++jj) {
                    res[jj] += sv[jj];
                }
            }
            res.store(global_accumulator + res_idx * Ev + e);
        }

        lse_target[res_idx] = own_lse;
    }
}

template<int Ev, class scalar_t>
__global__ __launch_bounds__(32) void hogwild_attention_reduce_kernel(
        scalar_t* out, const float* v_buffer, const float* lse_buffer, int splits, Shape shape) {
    int h = blockIdx.x;
    int w = blockIdx.y % shape.W;
    int s = blockIdx.y / shape.W;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    using v_cache_t = GenericVector<float, Ev / warp.size()>;
    v_cache_t v_cache = v_cache_t::zeros();

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;

    float own_lse = std::numeric_limits<float>::lowest();

    for (int split = 0; split < splits; ++split) {
        int res_idx = ((split * shape.W + w) * shape.Hq + h) * shape.S + s;
        const float* split_res = v_buffer + res_idx * Ev;
        float res_lse = lse_buffer[res_idx];
        if (res_lse == std::numeric_limits<float>::lowest()) {
            continue;
        }
        float max = std::max(own_lse, res_lse);
        float sa = std::exp2f(own_lse - max);
        float sb = std::exp2f(res_lse - max);
        float rescaler_a = sa / (sa + sb);
        float rescaler_b = sb / (sa + sb);
        #pragma unroll
        for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
            int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
            fvec_t sv = fvec_t::load_lu(split_res + e);
            for (int jj = 0; jj < fvec_t::size; ++jj) {
                float old = v_cache[ee + jj];
                float upd = old * rescaler_a + sv[jj] * rescaler_b;
                v_cache[ee + jj] = upd;
            }
        }
        own_lse = std::log2(sa + sb) + max;
    }

    for (int ee = 0; ee < Ev / warp.size(); ee += fvec_t::size) {
        int e = ee * warp.size() + warp.thread_rank() * fvec_t::size;
        vec_t st = vec_t::zeros();
        for (int jj = 0; jj < fvec_t::size; ++jj) {
            st[jj] = (scalar_t)v_cache[ee + jj];
        }
        st.store(out + ((w * shape.Hq + h) * shape.S + s) * Ev + e);
    }
}

template<class scalar_t>
cudaError_t hogwild_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const int* fragment_lengths,
                           const scalar_t** key_fragments,
                           const scalar_t** value_fragments,
                           Shape shape) {
    int problem_size = shape.Hkv * shape.W * shape.S;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // Note: The current kernel will **not** work if there is only one split!
    int splits = max(2, sms / problem_size);

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)shape.W * (unsigned)shape.S, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = shape.Ev * sizeof(float) * block_dim.x / 32 * (shape.Hq / shape.Hkv);
    smem += 2 * sizeof(float) * (shape.Hq / shape.Hkv);
    smem = std::max(smem, 2 * (shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));
    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    std::size_t required_workspace = shape.W * shape.Hq * shape.S * splits;  // [W, Hq, S, K]
    size_t alloc = required_workspace * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_workspace) {
        if (workspace)
            CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, alloc));
        CUDA_RETURN_ON_ERROR(cudaMemset(workspace, 0, alloc));
        workspace_size = required_workspace;
    }

    if (shape.E == 128 && shape.Ev == 128 && shape.Hq == shape.Hkv * 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(hogwild_attention_gpu_kernel21<128, 128, 16, scalar_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        hogwild_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(
                out, workspace, scale, locations, queries, fragment_lengths, key_fragments, value_fragments, shape);

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.W * (unsigned)shape.S, 1};
        hogwild_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, (float*)workspace, (float*)workspace + splits * shape.W * shape.Hq * shape.S * shape.Ev,
                splits, shape);
    } else {
        printf("Unsupported head dimension");
    }
    return cudaGetLastError();
}

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void hogwild_paged_attention_gpu_kernel21(
        scalar_t* out, char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const int* fragment_lengths, // Corresponds to sequence lengths
        const scalar_t* key_cache,
        const scalar_t* value_cache,
        const int* block_table,
        Shape shape) {
    // Input:   key_cache/value_cache: [NumBlocks, Hkv, BlockSize, E/Ev]
    //          block_table: [W, max_blocks_per_seq]  (W is batch size)
    //          queries: [F, W, Hq, S, E]
    //          locations [F, W, S]
    // Output:  [W, Hq, S, Ev]

    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S;
    int block_size = shape.block_size;
    int max_blocks = shape.max_blocks_per_seq;
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    ptrdiff_t q_stride = E * S * Hq * W;
    extern __shared__ float scratch[];

    // adjust scale so we can use base-2 exponent later on
    float l2scale = scale / std::log(2.f);

    int hkv = blockIdx.x;
    int w = blockIdx.y % W;
    int s = blockIdx.y / W;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int hq = hkv * GQA;
    ptrdiff_t q_offset = ((w * Hq + hq) * S + s) * E;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);   // vectors per head per thread
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);  // vectors per head per thread

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using full_fvec_t = GenericVector<float, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>;
    qk_cache_t q_cache[GQA];

    // combine values
    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    // determine maximum and online logsumexp
    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

        for (int f = 0; f < shape.F; ++f) {
            int q_loc = locations[(f * W + w) * S + s];
            int L = fragment_lengths[f];
            int maxL = std::min(L, q_loc + 1);

            for (int gqa = 0; gqa < GQA; ++gqa) {
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    full_vec_t qv = full_vec_t::load(queries + f * q_stride + q_offset + gqa * S * E + e);
                    for (int j = 0; j < VecSize; ++j) {
                        q_cache[gqa][ee * VecSize + j] = qv[j];
                    }
                }
            }

            // MODIFICATION: Use block_table to calculate addresses instead of fragment pointers
            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;

                // Paged attention address calculation
                int block_idx_in_seq = l / block_size;
                int offset_in_block = l % block_size;
                int physical_block_id = block_table[w * max_blocks + block_idx_in_seq];

                ptrdiff_t k_block_start = (ptrdiff_t)physical_block_id * shape.Hkv * block_size * E;
                ptrdiff_t v_block_start = (ptrdiff_t)physical_block_id * shape.Hkv * block_size * Ev;

                ptrdiff_t k_offset = k_block_start + (hkv * block_size + offset_in_block) * E;
                ptrdiff_t v_offset = v_block_start + (hkv * block_size + offset_in_block) * Ev;

                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x,
                                            key_cache + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x,
                                            value_cache + v_offset + e, sizeof(full_vec_t));
                }
            };

            int stage = 0;
            ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
            __pipeline_commit();
            ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
            __pipeline_commit();

            for (int ll = split; ll < maxL; ll += StepSize) {
                // This part of the loop is identical to the original kernel
                int l = ll + sub_warp.meta_group_rank() * splits;
                qk_cache_t keys;
                v_cache_t vals;
                __pipeline_wait_prior(1);
                if (l >= maxL) continue;
                unsigned mask = __activemask();

                for (int ee = 0; ee < VPH_k; ++ee) {
                    full_vec_t tmp = keys_lookahead[(stage * VPH_k + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) {
                        keys[ee * VecSize + j] = (float)tmp[j];
                    }
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    full_vec_t tmp = vals_lookahead[(stage * VPH_v + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) {
                        vals[ee * VecSize + j] = (float)tmp[j];
                    }
                }

                ldg_sts((stage + 2) % 2, l + 2 * StepSize);
                stage = (stage + 1) % 2;
                __pipeline_commit();

                float qk[GQA] = {};
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    for (int ee = 0; ee < VPH_k; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            qk[gqa] += q_cache[gqa][ee * VecSize + j] * keys[ee * VecSize + j];
                        }
                    }
                }

                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0100, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0010, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0001, 8);
                }

                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    if (qk[gqa] > maximum[gqa]) {
                        float rescale = std::exp2f(l2scale * (maximum[gqa] - qk[gqa]));
                        for (int j = 0; j < v_cache_t::size; ++j) {
                            v_cache[gqa][j] *= rescale;
                        }
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));

                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) {
                            v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                        }
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    // The reduction part is identical to the original kernel
    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.thread_rank() == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + 2 * sub_warp.meta_group_rank() + 2 * WarpSize * gqa);
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        float r_max = maximum[gqa];
        float l_max = maximum[gqa];
        float r_lse = 0;
        if (warp.thread_rank() < SubWarpMetaSize) {
            stats_t data = stats_t::load(scratch + 2 * warp.thread_rank() + 2 * WarpSize * gqa);
            r_max = data[0];
            r_lse = data[1];
        }

        maximum[gqa] = cg::reduce(warp, r_max, cg::greater<float>{});
        r_lse *= std::exp2f(l2scale * (r_max - maximum[gqa]));
        lse[gqa] = cg::reduce(warp, r_lse, cg::plus<float>{});

        if (lse[gqa] != 0) {
            float rescale = std::exp2f(l2scale * (l_max - maximum[gqa])) / lse[gqa];
            for (int j = 0; j < v_cache_t::size; ++j) {
                v_cache[gqa][j] *= rescale;
            }
        }

        if (threadIdx.x == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        }

        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_cache[gqa][ee * VecSize + j];
                static_assert(SubWarpSize == 8);
                v += __shfl_xor_sync(0xffffffff, v, 0b10000, WarpSize);
                v += __shfl_xor_sync(0xffffffff, v, 0b01000, WarpSize);
                v_cache[gqa][ee * VecSize + j] = v;
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.meta_group_rank() % (WarpSize / SubWarpSize) == 0) {
            for (int ee = 0; ee < VPH_v; ++ee) {
                int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                full_fvec_t store;
                for (int j = 0; j < VecSize; ++j) {
                    store[j] = v_cache[gqa][ee * VecSize + j];
                }
                store.store(scratch + e + Ev * sub_warp.meta_group_rank() / (WarpSize / SubWarpSize) + gqa * 256 / WarpSize * Ev);
            }
        }
    }
    __syncthreads();

    for (int gqa_offset = 0; gqa_offset < GQA; gqa_offset += (blockDim.x / WarpSize)) {
        int gqa = warp.meta_group_rank() + gqa_offset;
        if (gqa >= GQA) {
            continue;
        }

        int h = hkv * GQA + gqa;
        int res_base = ((w * Hq + h) * S + s);
        int res_inc = W * Hq * S;
        int res_idx = res_base + split * res_inc;
        float* global_accumulator = reinterpret_cast<float*>(workspace);
        float* lse_target = global_accumulator + W * Hq * S * Ev * splits;

        stats_t data = stats_t::load(scratch + GQA * 256 / WarpSize * Ev + gqa * 2);
        float own_lse = data[1];
        float own_max = data[0];
        own_lse = std::log2(own_lse) + l2scale * own_max;

        for (int e = vec_t::size * warp.thread_rank(); e < Ev; e += vec_t::size * warp.size()) {
            fvec_t res = fvec_t::zeros();
            for (int j = 0; j < SubWarpMetaSize / (WarpSize / SubWarpSize); ++j) {
                fvec_t sv = fvec_t::load(scratch + e + Ev * j + gqa * 256 / WarpSize * Ev);
                for (int jj = 0; jj < vec_t::size; ++jj) {
                    res[jj] += sv[jj];
                }
            }
            res.store(global_accumulator + res_idx * Ev + e);
        }

        lse_target[res_idx] = own_lse;
    }
}

template<class scalar_t>
cudaError_t hogwild_paged_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const int* fragment_lengths,
                           const scalar_t* key_cache,
                           const scalar_t* value_cache,
                           const int* block_table,
                           Shape shape) {
    int problem_size = shape.Hkv * shape.W * shape.S;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    // Note: The current kernel will **not** work if there is only one split!
    int splits = max(2, sms / problem_size);

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)shape.W * (unsigned)shape.S, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = shape.Ev * sizeof(float) * block_dim.x / 32 * (shape.Hq / shape.Hkv);
    smem += 2 * sizeof(float) * (shape.Hq / shape.Hkv);
    smem = std::max(smem, 2 * (shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));
    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    std::size_t required_workspace = shape.W * shape.Hq * shape.S * splits;  // [W, Hq, S, K]
    size_t alloc = required_workspace * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < required_workspace) {
        if (workspace)
            CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, alloc));
        CUDA_RETURN_ON_ERROR(cudaMemset(workspace, 0, alloc));
        workspace_size = required_workspace;
    }

    if (shape.E == 128 && shape.Ev == 128 && shape.Hq == shape.Hkv * 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(hogwild_paged_attention_gpu_kernel21<128, 128, 16, scalar_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        hogwild_paged_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(
                out, workspace, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table, shape);

        dim3 r_grid_dim{(unsigned)shape.Hq, (unsigned)shape.W * (unsigned)shape.S, 1};
        hogwild_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, (float*)workspace, (float*)workspace + splits * shape.W * shape.Hq * shape.S * shape.Ev,
                splits, shape);
    } else {
        printf("Unsupported head dimension");
    }
    return cudaGetLastError();
}

template<int Ev, class scalar_t>
__global__ __launch_bounds__(32) void hogwild_varlen_attention_reduce_kernel(
        scalar_t* out, const float* v_buffer, const float* lse_buffer, int splits,
        int total_q_tokens, int Hq) {

    // Each block reduces one (token, head) output.
    int q_token_idx = blockIdx.x;
    int h = blockIdx.y;

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Local registers to hold the final accumulated result.
    // Each thread in the warp handles a slice of the Ev dimension.
    constexpr int VELEMS = Ev / warp.size();
    float v_cache[VELEMS];
    #pragma unroll
    for (int i = 0; i < VELEMS; ++i) {
        v_cache[i] = 0.0f;
    }

    float own_lse = -std::numeric_limits<float>::infinity();

    // Iterate over all splits to reduce the partial results.
    for (int split = 0; split < splits; ++split) {
        // Calculate the base index for the current (split, token, head).
        // v_buffer layout: [splits, total_q, Hq, Ev]
        // lse_buffer layout: [splits, total_q, Hq]
        ptrdiff_t base_idx = ((ptrdiff_t)split * total_q_tokens + q_token_idx) * Hq + h;
        const float* split_res_v = v_buffer + base_idx * Ev;
        float split_res_lse = lse_buffer[base_idx];

        // Skip this split if it contributed nothing (e.g., all masked out).
        if (split_res_lse == -std::numeric_limits<float>::infinity()) {
            continue;
        }

        // Online softmax combination logic.
        float max_lse = fmaxf(own_lse, split_res_lse);

        // Avoid division by zero if both LSEs are -inf.
        if (max_lse == -std::numeric_limits<float>::infinity()) {
            continue;
        }

        float sa = exp2f(own_lse - max_lse);
        float sb = exp2f(split_res_lse - max_lse);

        float sum_sa_sb = sa + sb;
        if (sum_sa_sb == 0.0f) {
             continue;
        }

        float rescaler_a = sa / sum_sa_sb;
        float rescaler_b = sb / sum_sa_sb;

        // Update the value vector in v_cache.
        #pragma unroll
        for (int i = 0; i < VELEMS; ++i) {
            int e = i * warp.size() + warp.thread_rank();
            float v_in = __ldcs(split_res_v + e); // Use cache-bypassing load
            v_cache[i] = v_cache[i] * rescaler_a + v_in * rescaler_b;
        }

        // Update the LSE.
        own_lse = log2f(sum_sa_sb) + max_lse;
    }

    // Write the final result to the output tensor.
    // out layout: [total_q_tokens, Hq, Ev] -> flatten to [total_q_tokens, Hq * Ev]
    ptrdiff_t out_base_idx = ((ptrdiff_t)q_token_idx * Hq + h) * Ev;
    #pragma unroll
    for (int i = 0; i < VELEMS; ++i) {
        int e = i * warp.size() + warp.thread_rank();
        out[out_base_idx + e] = (scalar_t)v_cache[i];
    }
}

template<int E, int Ev, int GQA, class scalar_t>
__global__ __launch_bounds__(256) void hogwild_varlen_paged_attention_gpu_kernel21(
        char* workspace, float scale,
        const int* locations, const scalar_t* queries,
        const int* fragment_lengths, // Corresponds to sequence lengths
        const scalar_t* key_cache,
        const scalar_t* value_cache,
        const int* block_table,
        Shape shape) {

    int W = shape.W;
    int Hq = shape.Hq;
    int block_size = shape.block_size;
    int max_blocks = shape.max_blocks_per_seq;
    assert(E == shape.E);
    assert(Ev == shape.Ev);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    auto sub_warp = cg::tiled_partition<SubWarpSize>(block);
    constexpr const int SubWarpMetaSize = 256 / SubWarpSize;

    extern __shared__ float scratch[];

    float l2scale = scale / std::log(2.f);

    // --- VARLEN INDEXING ---
    int hkv = blockIdx.x;
    int q_token_idx = blockIdx.y;
    int split = blockIdx.z;
    int splits = gridDim.z;

    int w = 0;
    while (w < shape.W && q_token_idx >= shape.cu_seqlens_q[w + 1]) {
        w++;
    }
    // s (local pos in sequence) is not explicitly needed for indexing, but good for context
    // int s = q_token_idx - shape.cu_seqlens_q[w];
    // --- END VARLEN INDEXING ---

    int hq = hkv * GQA;
    ptrdiff_t q_stride_f = (ptrdiff_t)gridDim.y * Hq * E;
    ptrdiff_t q_offset = ((ptrdiff_t)q_token_idx * Hq + hq) * E;

    constexpr const int VecSize = 16 / sizeof(scalar_t);
    constexpr int VPH_k = E / (SubWarpSize * VecSize);
    constexpr int VPH_v = Ev / (SubWarpSize * VecSize);

    using full_vec_t = GenericVector<scalar_t, VecSize>;
    using qk_cache_t = GenericVector<float, E / SubWarpSize>;
    qk_cache_t q_cache[GQA];

    using v_cache_t = GenericVector<float, Ev / SubWarpSize>;
    v_cache_t v_cache[GQA];
    float maximum[GQA];
    for (int gqa = 0; gqa < GQA; ++gqa) {
        v_cache[gqa] = v_cache_t::zeros();
        maximum[gqa] = std::numeric_limits<float>::lowest();
    }

    float lse[GQA] = {};
    {
        full_vec_t* keys_lookahead = reinterpret_cast<full_vec_t*>(scratch);
        full_vec_t* vals_lookahead = keys_lookahead + 2 * VPH_k * 256;

        for (int f = 0; f < shape.F; ++f) {
            int q_loc = locations[(ptrdiff_t)f * gridDim.y + q_token_idx];
            int L = fragment_lengths[f];
            int maxL = std::min(L, q_loc + 1);

            for (int gqa = 0; gqa < GQA; ++gqa) {
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    full_vec_t qv = full_vec_t::load(queries + f * q_stride_f + q_offset + (ptrdiff_t)gqa * E + e);
                    for (int j = 0; j < VecSize; ++j) {
                        q_cache[gqa][ee * VecSize + j] = qv[j];
                    }
                }
            }

            const int StepSize = SubWarpMetaSize * splits;
            auto ldg_sts = [&](int stage, int l) {
                if (l >= maxL) return;
                int block_idx_in_seq = l / block_size;
                int offset_in_block = l % block_size;
                int physical_block_id = block_table[(ptrdiff_t)w * max_blocks + block_idx_in_seq];
                ptrdiff_t k_block_start = (ptrdiff_t)physical_block_id * shape.Hkv * block_size * E;
                ptrdiff_t v_block_start = (ptrdiff_t)physical_block_id * shape.Hkv * block_size * Ev;
                ptrdiff_t k_offset = k_block_start + ((ptrdiff_t)hkv * block_size + offset_in_block) * E;
                ptrdiff_t v_offset = v_block_start + ((ptrdiff_t)hkv * block_size + offset_in_block) * Ev;
                for (int ee = 0; ee < VPH_k; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(keys_lookahead + (stage * VPH_k + ee) * 256 + threadIdx.x, key_cache + k_offset + e, sizeof(full_vec_t));
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    int e = (ee * SubWarpSize + sub_warp.thread_rank()) * VecSize;
                    __pipeline_memcpy_async(vals_lookahead + (stage * VPH_v + ee) * 256 + threadIdx.x, value_cache + v_offset + e, sizeof(full_vec_t));
                }
            };

            int stage = 0;
            ldg_sts(0, sub_warp.meta_group_rank() * splits + split);
            __pipeline_commit();
            ldg_sts(1, sub_warp.meta_group_rank() * splits + split + StepSize);
            __pipeline_commit();

            for (int ll = split; ll < maxL; ll += StepSize) {
                int l = ll + sub_warp.meta_group_rank() * splits;
                qk_cache_t keys;
                v_cache_t vals;
                __pipeline_wait_prior(1);
                if (l >= maxL) continue;
                unsigned mask = __activemask();
                for (int ee = 0; ee < VPH_k; ++ee) {
                    full_vec_t tmp = keys_lookahead[(stage * VPH_k + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) keys[ee * VecSize + j] = (float)tmp[j];
                }
                for (int ee = 0; ee < VPH_v; ++ee) {
                    full_vec_t tmp = vals_lookahead[(stage * VPH_v + ee) * 256 + threadIdx.x];
                    for (int j = 0; j < VecSize; ++j) vals[ee * VecSize + j] = (float)tmp[j];
                }
                ldg_sts((stage + 2) % 2, l + 2 * StepSize);
                stage = (stage + 1) % 2;
                __pipeline_commit();
                float qk[GQA] = {};
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    for (int ee = 0; ee < VPH_k; ++ee) {
                        for (int j = 0; j < VecSize; ++j) qk[gqa] += q_cache[gqa][ee * VecSize + j] * keys[ee * VecSize + j];
                    }
                }
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0100, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0010, 8);
                    qk[gqa] += __shfl_xor_sync(mask, qk[gqa], 0b0001, 8);
                }
                #pragma unroll
                for (int gqa = 0; gqa < GQA; ++gqa) {
                    if (qk[gqa] > maximum[gqa]) {
                        float rescale = std::exp2f(l2scale * (maximum[gqa] - qk[gqa]));
                        for (int j = 0; j < v_cache_t::size; ++j) v_cache[gqa][j] *= rescale;
                        lse[gqa] *= rescale;
                        maximum[gqa] = qk[gqa];
                    }
                    float att = std::exp2f(l2scale * (qk[gqa] - maximum[gqa]));
                    lse[gqa] += att;
                    for (int ee = 0; ee < VPH_v; ++ee) {
                        for (int j = 0; j < VecSize; ++j) v_cache[gqa][ee * VecSize + j] += att * vals[ee * VecSize + j];
                    }
                }
            }
            __pipeline_wait_prior(0);
        }
    }

    using vec_t = GenericVector<scalar_t, 4>;
    using fvec_t = GenericVector<float, 4>;
    using stats_t = GenericVector<float, 2>;

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.thread_rank() == 0) {
            stats_t data;
            data[0] = maximum[gqa];
            data[1] = lse[gqa];
            data.store(scratch + 2 * sub_warp.meta_group_rank() + 2 * WarpSize * gqa);
        }
    }

    __syncthreads();
    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        float r_max = -std::numeric_limits<float>::infinity();
        float r_lse = 0;
        if (warp.thread_rank() < SubWarpMetaSize) {
            stats_t data = stats_t::load(scratch + 2 * warp.thread_rank() + 2 * WarpSize * gqa);
            r_max = data[0];
            r_lse = data[1];
        }

        float block_max = cg::reduce(warp, r_max, cg::greater<float>{});
        if (r_max > -std::numeric_limits<float>::infinity()) {
            r_lse *= std::exp2f(l2scale * (r_max - block_max));
        }
        float block_lse = cg::reduce(warp, r_lse, cg::plus<float>{});

        if (block_lse != 0) {
            float rescale = std::exp2f(l2scale * (maximum[gqa] - block_max)) / block_lse;
            for (int j = 0; j < v_cache_t::size; ++j) v_cache[gqa][j] *= rescale;
        } else {
            for (int j = 0; j < v_cache_t::size; ++j) v_cache[gqa][j] = 0.0f;
        }

        size_t stats_storage_offset = (size_t)GQA * SubWarpMetaSize * Ev;
        if (threadIdx.x == 0) {
            stats_t data;
            data[0] = block_max;
            data[1] = block_lse;
            data.store(scratch + stats_storage_offset + gqa * 2);
        }

        for (int ee = 0; ee < VPH_v; ++ee) {
            for (int j = 0; j < VecSize; ++j) {
                float v = v_cache[gqa][ee * VecSize + j];
                v += __shfl_down_sync(0xffffffff, v, 4, 8);
                v += __shfl_down_sync(0xffffffff, v, 2, 8);
                v += __shfl_down_sync(0xffffffff, v, 1, 8);
                v_cache[gqa][ee * VecSize + j] = v;
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int gqa = 0; gqa < GQA; ++gqa) {
        if (sub_warp.thread_rank() == 0) {
            for (int ee = 0; ee < VPH_v; ++ee) {
                int e_base = ee * VecSize;
                using f_vec_t = GenericVector<float, VecSize>;
                f_vec_t to_store;
                for(int i=0; i<VecSize; ++i) to_store[i] = v_cache[gqa][e_base + i];
                float* store_addr = scratch + (gqa * SubWarpMetaSize + sub_warp.meta_group_rank()) * Ev + e_base;
                to_store.store(store_addr);
            }
        }
    }
    __syncthreads();

    for (int gqa_offset = 0; gqa_offset < GQA; gqa_offset += (blockDim.x / WarpSize)) {
        int gqa = warp.meta_group_rank() + gqa_offset;
        if (gqa >= GQA) continue;

        int h = hkv * GQA + gqa;
        ptrdiff_t res_idx = (ptrdiff_t)split * gridDim.y + q_token_idx;
        float* global_accumulator = reinterpret_cast<float*>(workspace);
        float* lse_target = global_accumulator + (size_t)splits * gridDim.y * shape.Hq * Ev;

        size_t stats_storage_offset = (size_t)GQA * SubWarpMetaSize * Ev;
        stats_t block_stats = stats_t::load(scratch + stats_storage_offset + gqa * 2);
        float block_max = block_stats[0];
        float block_lse = block_stats[1];

        if (block_lse > 0) {
            lse_target[res_idx * shape.Hq + h] = log2f(block_lse) + l2scale * block_max;
        } else {
            lse_target[res_idx * shape.Hq + h] = -std::numeric_limits<float>::infinity();
        }

        for (int e = warp.thread_rank() * 4; e < Ev; e += warp.size() * 4) {
            GenericVector<float, 4> final_v = GenericVector<float, 4>::zeros();
            #pragma unroll
            for (int i = 0; i < SubWarpMetaSize; ++i) {
                GenericVector<float, 4> partial_v = GenericVector<float, 4>::load(scratch + (gqa * SubWarpMetaSize + i) * Ev + e);
                for(int j=0; j<4; ++j) final_v[j] += partial_v[j];
            }
            final_v.store(global_accumulator + (res_idx * shape.Hq + h) * Ev + e);
        }
    }
}

template<class scalar_t>
cudaError_t hogwild_varlen_paged_attention_gpu(scalar_t* out, float scale,
                           const int* locations, const scalar_t* queries,
                           const int* fragment_lengths,
                           const scalar_t* key_cache,
                           const scalar_t* value_cache,
                           const int* block_table,
                           Shape shape) {
    const int total_q_tokens = shape.cu_seqlens_q[shape.W];

    int problem_size = shape.Hkv * total_q_tokens;
    int sms = -1;
    CUDA_RETURN_ON_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int splits = max(2, sms > 0 && problem_size > 0 ? sms / problem_size : 2);

    dim3 grid_dim{(unsigned)shape.Hkv, (unsigned)total_q_tokens, (unsigned)splits};
    dim3 block_dim{256, 1, 1};
    size_t smem = (size_t)shape.Hq / shape.Hkv * SubWarpMetaSize * shape.Ev * sizeof(float);
    smem += (size_t)shape.Hq / shape.Hkv * 2 * sizeof(float);
    smem = std::max(smem, 2 * (size_t)(shape.E + shape.Ev) * (block_dim.x / SubWarpSize) * sizeof(scalar_t));
    static char* workspace = nullptr;
    static std::size_t workspace_size = 0;

    size_t required_workspace = (size_t)splits * total_q_tokens * shape.Hq;
    size_t alloc = required_workspace * (shape.Ev + 1) * sizeof(float);
    if (workspace_size < alloc) {
        if (workspace) CUDA_RETURN_ON_ERROR(cudaFree(workspace));
        CUDA_RETURN_ON_ERROR(cudaMalloc(&workspace, alloc));
        workspace_size = alloc;
    }

    if (shape.E == 128 && shape.Ev == 128 && shape.Hq == shape.Hkv * 16) {
        CUDA_RETURN_ON_ERROR(cudaFuncSetAttribute(hogwild_varlen_paged_attention_gpu_kernel21<128, 128, 16, scalar_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        hogwild_varlen_paged_attention_gpu_kernel21<128, 128, 16><<<grid_dim, block_dim, smem>>>(workspace, scale, locations, queries, fragment_lengths, key_cache, value_cache, block_table, shape);
        CUDA_CHECK_THROW(cudaGetLastError());

        const float* v_buffer = reinterpret_cast<const float*>(workspace);
        const float* lse_buffer = v_buffer + (size_t)splits * total_q_tokens * shape.Hq * shape.Ev;
        dim3 r_grid_dim{(unsigned)total_q_tokens, (unsigned)shape.Hq, 1};
        hogwild_varlen_attention_reduce_kernel<128><<<r_grid_dim, 32>>>(
                out, v_buffer, lse_buffer, splits, total_q_tokens, shape.Hq);
        CUDA_CHECK_THROW(cudaGetLastError());
    } else {
        printf("Unsupported head dimension for varlen");
    }
    return cudaSuccess;
}

}  // namespace v21
