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
// Configure execution without changing the legacy LtCreate signature.
int LtSetNumThreads(LtModel model, int threads);
void LtFree(LtModel model);
int LtFit(LtModel model, const float* x, size_t rows, size_t cols,
          const float* y);
// out has capacity rows floats and must not overlap x.
int LtPredict(LtModel model, const float* x, size_t rows, size_t cols,
              float* out);
size_t LtLossCount(LtModel model);
int LtLoss(LtModel model, double* out, size_t count);
// Read-only inspection. format: 0=text, 1=json; tree_index=-1 means all trees.
// names are optional NUL-terminated UTF-8 labels, in column order.
// Query with out=NULL, capacity=0. required includes the trailing NUL.
// Otherwise capacity must be >= required. Errors preserve out; a too-small
// buffer still receives the required size. Callers must not refit/close between
// query and write. No allocated string ownership crosses the ABI.
int LtDumpModel(LtModel model, int format, int tree_index,
                const char* const* names, size_t name_count, char* out,
                size_t capacity, size_t* required);
// x points to one sample with cols features; binary prediction is P(class=1).
// The output/query contract is identical to LtDumpModel.
int LtExplainPrediction(LtModel model, const float* x, size_t cols, int format,
                        const char* const* names, size_t name_count, char* out,
                        size_t capacity, size_t* required);
#ifdef __cplusplus
}
#endif
#endif  // LIBTREE_C_API_H_
