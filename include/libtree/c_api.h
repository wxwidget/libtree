#ifndef LIBTREE_C_API_H_
#define LIBTREE_C_API_H_
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Opaque owning handle; errors never escape across the C ABI.
typedef void* LtModel;
const char* LtLastError(void);
LtModel LtCreate(int trees, int depth, int bins, int min_leaf, double rate,
                 double l2, double min_gain, int binary);
void LtFree(LtModel model);
int LtFit(LtModel model, const float* x, size_t rows, size_t cols,
          const float* y);
int LtPredict(LtModel model, const float* x, size_t rows, size_t cols,
              float* out);
size_t LtLossCount(LtModel model);
int LtLoss(LtModel model, double* out, size_t count);
#ifdef __cplusplus
}
#endif
#endif  // LIBTREE_C_API_H_
