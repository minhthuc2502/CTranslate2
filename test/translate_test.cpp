#include <iostream>
#include <ctranslate2/translator.h>
#include <ctranslate2/generator.h>
#include <ctranslate2/models/language_model.h>
#include <ctranslate2/models/transformer.h>
#include <ctranslate2/layers/transformer.h>
#include <sentencepiece_processor.h>
#include <csignal>
#include <list>

#include "boost/program_options.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  ctranslate2::Translator translator("/home/pham/workplace/systran/openNMT/CTranslate2/test/models/enfr_model", ctranslate2::Device::CPU);
  const std::vector<std::vector<std::string>> batch = {{"Please", "say", "something", "fun", "for", "me"}};
  ctranslate2::TranslationOptions options;
  options.beam_size = 2;
  const std::vector<ctranslate2::TranslationResult> results = translator.translate_batch(batch);
  for (auto& result : results) {
    std::cout << result.output()[0];
  }
  return 1;
}
