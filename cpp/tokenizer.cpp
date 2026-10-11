#include "tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sora {

std::uint64_t Tokenizer::pair_key(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32U) | right;
}

Tokenizer::Tokenizer(std::vector<MergeRule> merges) : merges_(std::move(merges)) {
    for (std::size_t rank = 0; rank < merges_.size(); ++rank) {
        const auto& rule = merges_[rank];
        if (rule.merged < 256) {
            throw std::invalid_argument("merged token IDs must start at 256");
        }
        pair_to_merged_[pair_key(rule.left, rule.right)] = rule.merged;
        merge_rank_[rule.merged] = rank;
        expansions_[rule.merged] = {rule.left, rule.right};
        vocab_size_ = std::max(vocab_size_, rule.merged + 1);
    }
}

std::vector<std::uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<std::uint32_t> ids;
    ids.reserve(text.size());
    for (unsigned char byte : text) ids.push_back(static_cast<std::uint32_t>(byte));

    while (ids.size() >= 2) {
        std::size_t best_index = ids.size();
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::uint32_t best_merged = 0;

        for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
            const auto found = pair_to_merged_.find(pair_key(ids[i], ids[i + 1]));
            if (found == pair_to_merged_.end()) continue;
            const auto rank = merge_rank_.find(found->second);
            if (rank != merge_rank_.end() && rank->second < best_rank) {
                best_rank = rank->second;
                best_index = i;
                best_merged = found->second;
            }
        }

        if (best_index == ids.size()) break;
        ids[best_index] = best_merged;
        ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    return ids;
}

void Tokenizer::expand(std::uint32_t id, std::string& out) const {
    if (id < 256) {
        out.push_back(static_cast<char>(static_cast<unsigned char>(id)));
        return;
    }
    const auto it = expansions_.find(id);
    if (it == expansions_.end()) {
        throw std::invalid_argument("unknown token ID: " + std::to_string(id));
    }
    expand(it->second.first, out);
    expand(it->second.second, out);
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    std::string out;
    for (std::uint32_t id : ids) expand(id, out);
    return out;
}

std::uint32_t Tokenizer::vocab_size() const {
    return vocab_size_;
}

} // namespace sora
