#ifndef BOOSTED_TREE_H
#define BOOSTED_TREE_H
#include <vector>
#include "regression_tree.h"
class BoostedTree: public std::vector<RegressionTree*>{
public:
    BoostedTree(){}
    float classify (const Instance* const instance, double alpha) const;
    float classify (const DataSet& data, double alpha, vector<float>& preds) const;
};
#endif
