#include "ctranslate2/ops/add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Add::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Add");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

    void Add::operator()(const std::vector<StorageView>& inputs, StorageView& c) const {
      PROFILE("Add");
      size_t input_size = inputs.size();
      if (input_size < 2)
        throw std::invalid_argument("Input's size is smaller than 2. "
                                    "The actual size is " + std::to_string(input_size));
      DEVICE_AND_TYPE_DISPATCH(inputs[0].device(), inputs[0].dtype(), (compute<D, T>(inputs, c)));
    }
  }
}
