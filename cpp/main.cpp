#include "tokenizer.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string input = argc > 1 ? argv[1] : "Sora can learn.";

    // Tiny illustrative merges only. A JSON vocabulary loader is a follow-up.
    const std::vector<sora::MergeRule> merges = {
        {static_cast<std::uint32_t>('S'), static_cast<std::uint32_t>('o'), 256},
        {256, static_cast<std::uint32_t>('r'), 257},
        {257, static_cast<std::uint32_t>('a'), 258},
        {static_cast<std::uint32_t>(' '), static_cast<std::uint32_t>('c'), 259},
        {259, static_cast<std::uint32_t>('a'), 260},
        {260, static_cast<std::uint32_t>('n'), 261}
    };

    const sora::Tokenizer tokenizer(merges);
    const auto ids = tokenizer.encode(input);

    std::cout << "Input: " << input << "\nToken IDs:";
    for (const auto id : ids) std::cout << ' ' << id;
    std::cout << "\nDecoded: " << tokenizer.decode(ids) << '\n';
    std::cout << "Vocabulary size: " << tokenizer.vocab_size() << '\n';
    return 0;
}
