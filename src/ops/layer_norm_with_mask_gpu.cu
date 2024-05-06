#include "ctranslate2/ops/layer_norm.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace at {
  namespace native {

    // Forward declaration of the CUDA kernels.
    template <typename T, typename SizeT>
    __global__ void LayerNormForwardCUDAKernel(SizeT N,
                                               float eps,
                                               const T* X,
                                               T* Y);

  }
}

namespace ctranslate2 {
  namespace ops {

#define CUDA_NUM_THREADS 512

    template <Device D, typename T>
    void LayerNormWMask::compute(const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t,
                            StorageView& output) const {
      at::native::LayerNormForwardCUDAKernel<cuda::device_type<T>, cuda::index_t>
        <<<outer_size, CUDA_NUM_THREADS, 0, cuda::get_cuda_stream()>>>(
          axis_size,
          _epsilon,
          cuda::device_cast(input.data<T>()),
          cuda::device_cast(output.data<T>()));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNormWMask::compute<Device::CUDA, T>(const StorageView& input,       \
                                        const dim_t axis,               \
                                        const dim_t outer_size,         \
                                        const dim_t axis_size,          \
                                        const dim_t inner_size,         \
                                        StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

#include <cub/block/block_reduce.cuh>

namespace at {
  namespace native {

    template <typename T, typename SizeT>
    __global__ void LayerNormForwardCUDAKernel(SizeT N,
                                               float eps,
                                               const T* X,
                                               T* Y) {
      typedef cub::BlockReduce<float, CUDA_NUM_THREADS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage m_temp_storage;
      __shared__ typename BlockReduce::TempStorage v_temp_storage;
      __shared__ float s_mean;
      __shared__ float s_variance;

      const SizeT i = blockIdx.x;

      float num_elements_not_masked = 0;
      float sum1 = 0;
      float sum2 = 0;
      for (SizeT j = threadIdx.x; j < N; j += blockDim.x) {
        const SizeT index = i * N + j;
        sum1 += float(X[index]);
        sum2 += float(X[index]) * float(X[index]);
        num_elements_not_masked += 1;
      }
      sum1 = BlockReduce(m_temp_storage).Sum(sum1);
      sum2 = BlockReduce(v_temp_storage).Sum(sum2);
      num_elements_not_masked = BlockReduce(v_temp_storage).Sum(num_elements_not_masked);
      if (threadIdx.x == 0) {
        sum1 /= num_elements_not_masked;
      }
      for (SizeT j = threadIdx.x; j < N; j += blockDim.x) {
        const SizeT index = i * N + j;
        sum2 += (float(X[index]) - s_mean) * (float(X[index]) - s_mean);
      }
      sum2 = BlockReduce(v_temp_storage).Sum(sum2);
      if (threadIdx.x == 0) {
        sum2 /= num_elements_not_masked;
        s_mean = sum1;
        s_variance = rsqrtf(sum2 + eps);
      }

      __syncthreads();

      for (SizeT j = threadIdx.x; j < N; j += blockDim.x) {
        const SizeT index = i * N + j;
        Y[index] = (float(X[index]) - s_mean) * s_variance;
      }
    }

  }
}
