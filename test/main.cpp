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
      //auto user_tokens = sp.EncodeAsPieces(it->content);
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
  po::options_description cmdLineOptions("CTranslate2 Llama Options");
  po::variables_map vm;
  std::string model_path;
  std::string input_path;
  std::string system_prompt;
  int context_length;
  int generation_length;

  try
  {
    cmdLineOptions.add_options()
    ("help,h", "help message")
    ("model-path,p", po::value<std::string>(&model_path), "model path")
    ("input-path,i", po::value<std::string>(&input_path), "input path")
    ("system-prompt,s", po::value<std::string>(&system_prompt), "system prompt")
    ("context-length,c", po::value<int>(&context_length)->default_value(10000), "max context length")
    ("generation-length,g", po::value<int>(&generation_length)->default_value(512), "max generation length")
    ;

    po::positional_options_description p;
    p.add("model-path", 1);
    po::basic_parsed_options<char> bpo = po::command_line_parser(argc, argv).options(cmdLineOptions).positional(p).allow_unregistered().run();
    po::store(bpo, vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << cmdLineOptions << std::endl;
      return 0;
    }
  }
  catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  size_t max_prompt_length = context_length - generation_length;
  signal(SIGINT, signalHandler);
  std::cout << "Loading the model ..." << std::endl;

  ctranslate2::Generator generator(model_path, ctranslate2::Device::CPU);
  sentencepiece::SentencePieceProcessor sp_process;
  std::vector<std::string> token_buffer;
  ctranslate2::GenerationOptions generator_options;
  std::string generation_output;
  bool isFinished = false;
  generator_options.beam_size = 1;
  generator_options.max_length = generation_length;
  generator_options.sampling_temperature = 0.75;
  generator_options.sampling_topk = 20;
  generator_options.sampling_topp = 1;
  generator_options.include_prompt_in_result = false;
  generator_options.callback = [&sp_process, &token_buffer, &generation_output, &isFinished](ctranslate2::GenerationStepResult step_result) {
    std::string token = step_result.token;
    std::string word_complete;
    if (token.find("▁") == 0 && !token_buffer.empty())
    {
      sp_process.Decode(token_buffer, &word_complete);
      token_buffer.clear();
      std::cout << word_complete << " " << std::flush;
      if (!generation_output.empty())
        generation_output += ' ';
      generation_output += word_complete;
    }
    token_buffer.push_back(token);
    isFinished = step_result.is_last;
    return false;
  };
  sp_process.Load(model_path + "/tokenizer.model");

  std::list<Dialog> dialogs;
  if (!system_prompt.empty())
  {
    Dialog dialog = {system_prompt, Role::SYSTEM};
    dialogs.emplace_back(dialog);
  }

  try {
    while(true) {
      std::cout << "You: ";
      std::string user_request;
      std::getline(std::cin, user_request);
/*
    std::ostringstream user_request;
    std::ifstream inputFile(input_path);
    // Check if the file was successfully opened
    if (inputFile.is_open()) {
      std::string line;

      // Read and print each line of the file
      while (std::getline(inputFile, line)) {
        user_request << line << '\n';
      }

      // Close the file when done
      inputFile.close();
    } else {
      std::cerr << "Unable to open the file.\n";
    }
*/

      std::vector<std::string> input_tokens;

      Dialog user_dialog = {user_request, Role::USER};
      //Dialog user_dialog = {user_request.str(), Role::USER};
      dialogs.emplace_back(user_dialog);
      while (true) {
        input_tokens = build_prompt(sp_process, dialogs);
        if (input_tokens.size() <= max_prompt_length)
          break;
        if (!system_prompt.empty()) {
          dialogs.erase(std::next(dialogs.begin(), 1), std::next(dialogs.begin(), 2));
        }
        else {
          dialogs.erase(dialogs.begin(), std::next(dialogs.begin(), 1));
        }
      }
      auto results = generator.generate_batch_async({input_tokens}, generator_options);

      std::cout << "llama2: " << std::flush;
      //auto words = generateWords(sp_process, results);
      while (!isFinished) {}
      isFinished = false;
      if (!token_buffer.empty())
      {
        std::string word_complete;
        sp_process.Decode(token_buffer, &word_complete);
        token_buffer.clear();
        std::cout << word_complete << " " << std::flush;
        if (!generation_output.empty())
          generation_output += ' ';
        generation_output += word_complete;
      }
      Dialog outputDialog = {generation_output, Role::ASSISTANT};
      dialogs.emplace_back(outputDialog);
      generation_output.clear();
      std::cout << std::endl;
    }
  }
  catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}
