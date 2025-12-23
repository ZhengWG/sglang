#pragma once
#include <cstddef>
#include <cstdint>
#if __cplusplus < 202002L
#include <experimental/source_location>
using source_location = std::experimental::source_location;
#else
#include <source_location>
using source_location = std::source_location;
#endif
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace radix_tree_v2 {

using token_t = std::int32_t;
using token_vec_t = std::vector<token_t>;
using token_slice = std::span<const token_t>;
using NodeHandle = std::size_t;
using IOTicket = std::uint32_t;

inline void
_assert(bool condition, const char* message = "Assertion failed", source_location loc = source_location::current()) {
  if (!condition) [[unlikely]] {
    std::string msg = message;
    msg = msg + " at " + loc.file_name() + ":" + std::to_string(loc.line()) + " in " + loc.function_name();
    throw std::runtime_error(msg);
  }
}

}  // namespace radix_tree_v2
