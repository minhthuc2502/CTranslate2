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

enum class Role {
  SYSTEM,
  USER,
  ASSISTANT
};
struct Dialog {
  std::string content;
  Role role;
};

std::string B_INST = "[INST]";
std::string E_INST =  "[/INST]";
std::string B_SYS = "<<SYS>>\n";
std::string E_SYS = "\n<</SYS>>\n\n";

void signalHandler(int signum)
{
  exit(signum);
}

std::vector<std::string> generateWords(const sentencepiece::SentencePieceProcessor& sp_process, std::vector<std::future<ctranslate2::GenerationResult>>& results)
{
  std::vector<std::string> output;
  for (auto& step_result : results)
  {
    std::vector<std::vector<std::string>> tokens = step_result.get().sequences;
    std::string word_complete;
    std::vector<std::string> token_buffer;
    for (auto &v : tokens) {
      for (auto &w : v) {
        if (w.find("▁") == 0 && !token_buffer.empty())
        {
          sp_process.Decode(token_buffer, &word_complete);
          output.push_back(word_complete);
          token_buffer.clear();
          std::cout << word_complete << " ";
        }
        token_buffer.push_back(w);
      }
    }
    if (!token_buffer.empty())
    {
      sp_process.Decode(token_buffer, &word_complete);
      output.push_back(word_complete);
      token_buffer.clear();
      std::cout << word_complete << " ";
    }
  }
  return output;
}

std::vector<std::string> build_prompt(sentencepiece::SentencePieceProcessor& sp, std::list<Dialog>& dialogs) {
  std::vector<std::string> input_tokens;
  if (dialogs.begin()->role == Role::SYSTEM) {
    auto first_user_dialog = std::next(dialogs.begin(), 1);
    first_user_dialog->content = B_SYS + dialogs.begin()->content + E_SYS + first_user_dialog->content;
  }
  auto it = dialogs.begin()->role == Role::SYSTEM ? std::next(dialogs.begin(), 1) : dialogs.begin();

  for (; it != dialogs.end(); ++it) {
    if (it->role == Role::USER) {
      input_tokens.emplace_back("<s>");
      auto user_tokens = sp.EncodeAsPieces(B_INST + " " + it->content + " " + E_INST);
      input_tokens.insert(input_tokens.end(), user_tokens.begin(), user_tokens.end());
    }
    else if (it->role == Role::ASSISTANT) {
      auto user_tokens = sp.EncodeAsPieces(it->content);
      input_tokens.insert(input_tokens.end(), user_tokens.begin(), user_tokens.end());
      input_tokens.emplace_back("</s>");
    }
  }
  return input_tokens;
}

int main(int argc, char** argv) {
  ctranslate2::Translator translator("/home/pham/workplace/systran/openNMT/CTranslate2/test/models/enfr_model", ctranslate2::Device::CPU);
  const std::vector<std::vector<std::string>> batch = {{"Please", "say", "something", "fun", "for", "me"}, {"Hello", "world"}};
  const std::vector<ctranslate2::TranslationResult> results = translator.translate_batch(batch);
  for (auto& result : results) {
    std::cout << result.output()[0];
  }
  return 1;
}
