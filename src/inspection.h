#ifndef LIBTREE_INSPECTION_H_
#define LIBTREE_INSPECTION_H_

#include <ostream>
#include <string_view>

namespace libtree {
namespace internal {
inline void QuoteJson(std::ostream& output, std::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  output << '"';
  for (unsigned char c : text) {
    switch (c) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (c < 0x20)
          output << "\\u00" << kHex[c >> 4] << kHex[c & 15];
        else
          output << static_cast<char>(c);
    }
  }
  output << '"';
}
}  // namespace internal
}  // namespace libtree
#endif  // LIBTREE_INSPECTION_H_
