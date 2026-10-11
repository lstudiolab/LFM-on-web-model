#include "tokenizer.hpp"

#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace sora {
namespace {
std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open tokenizer file: " + path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
std::string json_string(const std::string& json, const std::string& key) {
    const auto key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) throw std::runtime_error("Tokenizer JSON is missing key: " + key);
    const auto colon = json.find(':', key_pos);
    const auto first = json.find('"', colon + 1);
    const auto last = json.find('"', first + 1);
    if (colon == std::string::npos || first == std::string::npos || last == std::string::npos)
        throw std::runtime_error("Malformed string field in tokenizer JSON: " + key);
    return json.substr(first + 1, last - first - 1);
}
std::uint32_t json_uint(const std::string& json, const std::string& key) {
    const auto key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) throw std::runtime_error("Tokenizer JSON is missing key: " + key);
    const auto colon = json.find(':', key_pos);
    if (colon == std::string::npos) throw std::runtime_error("Malformed JSON field: " + key);
    auto pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || !std::isdigit(static_cast<unsigned char>(json[pos])))
        throw std::runtime_error("Expected unsigned integer field: " + key);
    std::uint64_t value = 0;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        value = value * 10 + static_cast<unsigned>(json[pos] - '0');
        if (value > UINT32_MAX) throw std::runtime_error("Integer too large in tokenizer JSON: " + key);
        ++pos;
    }
    return static_cast<std::uint32_t>(value);
}
std::vector<std::uint32_t> parse_merges(const std::string& json) {
    const auto key_pos = json.find("\"merges\"");
    if (key_pos == std::string::npos) throw std::runtime_error("Tokenizer JSON is missing merges");
    const auto start = json.find('[', key_pos);
    if (start == std::string::npos) throw std::runtime_error("Malformed merges array");
    std::size_t pos = start;
    int depth = 0;
    bool in_string = false, escaped = false;
    std::vector<std::uint32_t> numbers;
    while (pos < json.size()) {
        const char c = json[pos];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            ++pos; continue;
        }
        if (c == '"') { in_string = true; ++pos; continue; }
        if (c == '[') { ++depth; ++pos; continue; }
        if (c == ']') { --depth; ++pos; if (depth == 0) break; continue; }
        if (depth > 0 && std::isdigit(static_cast<unsigned char>(c))) {
            std::uint64_t value = 0;
            while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
                value = value * 10 + static_cast<unsigned>(json[pos] - '0');
                if (value > UINT32_MAX) throw std::runtime_error("Merge ID too large");
                ++pos;
            }
            numbers.push_back(static_cast<std::uint32_t>(value));
            continue;
        }
        ++pos;
    }
    if (depth != 0 || numbers.size() % 3 != 0) throw std::runtime_error("Malformed BPE merges array");
    return numbers;
}
std::uint32_t codepoint_at(const std::string& s, std::size_t i, std::size_t& width) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { width = 1; return c; }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        width = 2; return ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        width = 3; return ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6)
                        | (static_cast<unsigned char>(s[i + 2]) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        width = 4; return ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12)
                        | ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6)
                        | (static_cast<unsigned char>(s[i + 3]) & 0x3F);
    }
    width = 1; return c;
}
bool unicode_space(std::uint32_t c) {
    return c == 0x20 || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0 || c == 0x1680
        || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F
        || c == 0x205F || c == 0x3000;
}
bool unicode_word(std::uint32_t c) {
    if (c == '_') return true;
    if (c < 0x80) return std::isalnum(static_cast<unsigned char>(c)) != 0;
    // Approximation of Python's Unicode \w categories without an external Unicode library.
    if ((c >= 0x2000 && c <= 0x206F) || (c >= 0x2190 && c <= 0x2BFF)
        || (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0x3000 && c <= 0x303F)
        || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0x1F3FB && c <= 0x1F3FF)) return false;
    return !unicode_space(c);
}
std::vector<std::string> split_pieces(const std::string& text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        // Python's training tokenizer makes each CR/LF line break a boundary,
        // separate from adjacent horizontal whitespace.
        if (text[i] == '\r' || text[i] == '\n') {
            const auto begin = i;
            if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') i += 2;
            else ++i;
            pieces.push_back(text.substr(begin, i - begin));
            continue;
        }

        std::size_t width = 1;
        const auto cp = codepoint_at(text, i, width);
        const int cat = unicode_space(cp) ? 0 : (unicode_word(cp) ? 1 : 2);
        const auto begin = i;
        i += width;
        while (i < text.size()) {
            if (text[i] == '\r' || text[i] == '\n') break;
            std::size_t next_width = 1;
            const auto next = codepoint_at(text, i, next_width);
            const int next_cat = unicode_space(next) ? 0 : (unicode_word(next) ? 1 : 2);
            if (next_cat != cat) break;
            i += next_width;
        }
        pieces.push_back(text.substr(begin, i - begin));
    }
    return pieces;
}
} // namespace

Tokenizer::Tokenizer(std::vector<MergeRule> merges) : merges_(std::move(merges)) {
    std::uint32_t expected = 256;
    for (const auto& rule : merges_) {
        if (rule.merged != expected || rule.left >= rule.merged || rule.right >= rule.merged)
            throw std::invalid_argument("Invalid BPE merge: IDs must be sequential and reference earlier tokens");
        expansions_[rule.merged] = {rule.left, rule.right};
        ++expected;
    }
    vocab_size_ = expected;
}
Tokenizer Tokenizer::load(const std::string& path) {
    const auto json = read_file(path);
    if (json_string(json, "format") != "sora-byte-bpe-v1") throw std::runtime_error("Unsupported tokenizer format");
    if (json_uint(json, "base_vocab_size") != 256) throw std::runtime_error("Tokenizer base vocabulary must be 256 bytes");
    const auto vocab_size = json_uint(json, "vocab_size");
    const auto values = parse_merges(json);
    std::vector<MergeRule> rules;
    rules.reserve(values.size() / 3);
    for (std::size_t i = 0; i < values.size(); i += 3) rules.push_back({values[i], values[i + 1], values[i + 2]});
    Tokenizer tokenizer(std::move(rules));
    if (tokenizer.vocab_size() != vocab_size) throw std::runtime_error("Tokenizer vocabulary size does not match merge table");
    return tokenizer;
}
std::vector<std::uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<std::uint32_t> output;
    for (const auto& piece : split_pieces(text)) {
        std::vector<std::uint32_t> ids;
        ids.reserve(piece.size());
        for (unsigned char byte : piece) ids.push_back(byte);
        for (const auto& rule : merges_) {
            if (ids.size() < 2) break;
            std::vector<std::uint32_t> next;
            next.reserve(ids.size());
            std::size_t i = 0;
            while (i < ids.size()) {
                if (i + 1 < ids.size() && ids[i] == rule.left && ids[i + 1] == rule.right) {
                    next.push_back(rule.merged); i += 2;
                } else next.push_back(ids[i++]);
            }
            ids = std::move(next);
        }
        output.insert(output.end(), ids.begin(), ids.end());
    }
    return output;
}
void Tokenizer::expand(std::uint32_t id, std::string& out) const {
    std::vector<std::uint32_t> pending{id};
    while (!pending.empty()) {
        const auto current = pending.back();
        pending.pop_back();
        if (current < 256) {
            out.push_back(static_cast<char>(static_cast<unsigned char>(current)));
            continue;
        }
        const auto it = expansions_.find(current);
        if (it == expansions_.end())
            throw std::invalid_argument("Unknown token ID: " + std::to_string(current));
        pending.push_back(it->second.second);
        pending.push_back(it->second.first);
    }
}
std::string Tokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    std::string out;
    for (const auto id : ids) expand(id, out);
    return out;
}
std::uint32_t Tokenizer::vocab_size() const { return vocab_size_; }
} // namespace sora
