#ifndef GRADIENT_BOOSTED_TREE_H 
#define GRADIENT_BOOSTED_TREE_H
#include <vector>
#include "regression_tree.h"
class GBDTree: public std::vector<RegressionTree*>{
public:
    GBDTree():T0(0){}
    ~GBDTree();
    float classify (const Instance* const instance) const;
    float classify (const DataSet& data, vector<float>& preds) const;
    float train(const DataSet& data, const Index& index, const args_t& myargs);
    void save(std::ostream&);
    void load(std::istream&); 
private:
    GBDTree(GBDTree&) {};
    float T0;
};
#endif
