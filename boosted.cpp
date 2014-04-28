#include "regression_tree.h"
double RegressionTree::boosted_classify(const vector<RegressionTree*>& bdt, const Instance* const instance, double alpha) {
    double r = 0.0;
    int nt = bdt.size();
    for(int i = 0; i < nt; i++) {
        r = r + alpha * bdt[i]->classify(instance);
    }
    return r;
}
