#include "ctranslate2/layers/encoder_classification.h"

namespace ctranslate2 {
  namespace layers {
    WiseAttentionLayer::WiseAttentionLayer(const models::Model &model, const std::string &scope)
    : _num_layers(model.get_attribute_with_default<int32_t >(scope + "/num_layers", 0))
    , _layer_norm(model.get_flag_with_default(scope + "/layer_norm", false))
    , _gam(model.get_variable(scope + "/gam"))
    , _scalar_parameters(model.get_variable(scope + "/scalar_parameters"))
    {
    }

    void WiseAttentionLayer::operator()(const ctranslate2::StorageView &input,
                                        ctranslate2::StorageView &output) {
      const Device device = input.device();
      const DataType dtype = input.dtype();
      StorageView normed_weights(dtype, device);
      ops::SoftMax()(_scalar_parameters, normed_weights);

      StorageView input_part_norm(dtype, device);
      StorageView result(dtype, device);
      for (dim_t i = 0; i < normed_weights.size(); ++i) {
        StorageView tmp(dtype, device);
        ops::Slide slide_ops(0, i, 1, true);
        StorageView normed_weight(dtype, device);
        StorageView input_part(dtype, device);
        slide_ops(normed_weights, normed_weight);
        slide_ops(input, input_part);
        input_part.reshape({input_part.dim(1), -1});

        if (_layer_norm) {
          ops::LayerNormWMask norm_ops(1e-12);
          norm_ops(input_part, input_part_norm);
          ops::Mul()(input_part_norm, normed_weight, tmp);
        }
        else {
          ops::Mul()(input_part, normed_weight, tmp);
        }
        if (result.empty()) {
          result = std::move(tmp);
        }
        else {
          ops::Add()(result, tmp, result);
        }
      }
      ops::Mul()(result, _gam, result);
      output = std::move(result);
    }

    EstimatorFeedForward::EstimatorFeedForward(const models::Model& model,
                                               const std::string& scope)
      : _activation_type(model.get_enum_value<ops::ActivationType>(scope + "/activation"))
      , _layers(build_layers_list<const Dense>(
        model,
        scope + "/layer",
        &_activation_type))
      , _layer_out(model, scope + "/linear_out", nullptr, true) {
    }

    void EstimatorFeedForward::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;

      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView inner(dtype, device);
      StorageView outter(dtype, device);
      for (auto &layer : _layers) {
        (*layer)(inner, outter);
        inner = std::move(outter);
      }
      _layer_out(inner, output);
    }
  }
}