#ifndef BOOSTED_TREE_H
#define BOOSTED_TREE_H
#include <vector>
#include "regression_tree.h"
class BoostedTree: public std::vector<RegressionTree*>{
public:
    BoostedTree(){}
    ~BoostedTree();
    float classify (const Instance* const instance) const;
    float classify (const DataSet& data, vector<float>& preds) const;
    float train(const DataSet& data, const Index& index, const args_t& myargs);
    void save(std::ostream&);
    void load(std::istream&); 
private:
    BoostedTree(BoostedTree&){};
};
#endif
