#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Add : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;
      void operator()(const std::vector<StorageView>& inputs, StorageView& c) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        c.resize_as(a);
        if (b.is_scalar()) {
          primitives<D>::add(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());
        } else {
          primitives<D>::add(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }

      template <Device D, typename T>
      void compute(const std::vector<StorageView>& inputs, StorageView& c) const {
        c.resize_as(inputs[0]);
        if (inputs[0].is_scalar()) {
          primitives<D>::add(inputs[0].data<T>()[0], inputs[1].data<T>(), c.data<T>(), c.size());
        } else {
          primitives<D>::add(inputs[0].data<T>(), inputs[1].data<T>(), c.data<T>(), c.size());
        }
        for (size_t i = 2; i < inputs.size(); ++i)
        {
          if (inputs[i].is_scalar()) {
            primitives<D>::add(c.data<T>()[0], inputs[i].data<T>(), c.data<T>(), c.size());
          } else {
            primitives<D>::add(c.data<T>(), inputs[i].data<T>(), c.data<T>(), c.size());
          }
        }
      }
    };

  }
}
