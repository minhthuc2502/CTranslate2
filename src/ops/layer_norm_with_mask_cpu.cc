#include "ctranslate2/ops/layer_norm.h"

#include "cpu/kernels.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNormWMask::compute(const StorageView& input,
                                const dim_t axis,
                                const dim_t outer_size,
                                const dim_t axis_size,
                                const dim_t inner_size,
                                StorageView& output) const {
/*
      CPU_ISA_DISPATCH((cpu::layer_norm_with_mask<ISA>(input.data<T>(),
                                                  output.data<T>(),
                                                  outer_size,
                                                  axis_size,
                                                  inner_size,
                                                  _epsilon)));
*/
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNormWMask::compute<Device::CPU, T>(const StorageView& input,        \
                                       const dim_t axis,                \
                                       const dim_t outer_size,          \
                                       const dim_t axis_size,           \
                                       const dim_t inner_size,          \
                                       StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
