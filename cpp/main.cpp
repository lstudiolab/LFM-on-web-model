#include "tokenizer.hpp"
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: sora-tokenizer <tokenizer.json> [text ...]\n";
        return 2;
    }
    try {
        const auto tokenizer = sora::Tokenizer::load(argv[1]);
        std::string text;
        for (int i = 2; i < argc; ++i) {
            if (i > 2) text += ' ';
            text += argv[i];
        }
        const auto ids = tokenizer.encode(text);
        std::cout << "Input: " << text << "\nIDs:";
        for (const auto id : ids) std::cout << ' ' << id;
        std::cout << "\nDecoded: " << tokenizer.decode(ids)
                  << "\nVocabulary size: " << tokenizer.vocab_size() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Tokenizer error: " << e.what() << '\n';
        return 1;
    }
}
