#include <sentencepiece_processor.h>
#include <ctranslate2/generator.h>


int main(int argc, char** argv) {
  ctranslate2::Generator generator("/home/pham/workplace/systran/openNMT/CTranslate2/test/models/enfr_model", ctranslate2::Device::CPU);
  const std::vector<std::vector<std::string>> batch = {{"▁[", "INST", "]", "▁What", "▁is", "▁your", "▁name", "?", "▁[", "/", "INST", "]"}};
  auto results = generator.generate_batch_async(batch);
  for (auto result : results) {

  }

  return 1;
}
