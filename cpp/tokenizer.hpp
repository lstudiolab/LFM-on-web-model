#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sora {
struct MergeRule { std::uint32_t left; std::uint32_t right; std::uint32_t merged; };

class Tokenizer {
public:
    static Tokenizer load(const std::string& path);
    explicit Tokenizer(std::vector<MergeRule> merges = {});
    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& ids) const;
    std::uint32_t vocab_size() const;
private:
    std::vector<MergeRule> merges_;
    std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> expansions_;
    std::uint32_t vocab_size_ = 256;
    void expand(std::uint32_t id, std::string& out) const;
};
} // namespace sora
