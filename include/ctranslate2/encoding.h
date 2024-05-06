#pragma once

#include <optional>

#include "storage_view.h"

namespace ctranslate2 {

  struct EncoderForwardOutput {
    StorageView last_hidden_state;

    std::optional<StorageView> pooler_output;
    std::optional<StorageView> wise_attn_output;
    std::optional<StorageView> estimator_ffn_output;
  };

}
