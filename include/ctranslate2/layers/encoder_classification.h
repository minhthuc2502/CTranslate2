#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {
    class WiseAttentionLayer : public Layer {
    public:
      WiseAttentionLayer(const models::Model& model,
                         const std::string& scope);

      void operator()(const StorageView& input,
                      StorageView& output);

      DataType output_type() const override {
        return _gam.dtype();
      }

      dim_t output_size() const override {
        return 0;
      }

    private:
      const dim_t _num_layers;
      const bool _layer_norm;
      const StorageView& _gam;
      const StorageView& _scalar_parameters;
    };

    class EstimatorFeedForward : public Layer
    {
    public:
      EstimatorFeedForward(const models::Model& model,
                         const std::string& scope);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _layer_out.output_type();
      }

      dim_t output_size() const override {
        return _layer_out.output_size();
      }

    private:
      const ops::ActivationType _activation_type;
      const std::vector<std::unique_ptr<const Dense>> _layers;
      const Dense _layer_out;
    };
  }
}