#include "boosted.h"
float BoostedTree::classify (const Instance* const instance, double alpha) const {
    double r = 0.0;
    const_iterator it = begin();
    for(; it != end(); ++it) {
        r = r + alpha * (*it)->classify(instance);
    }
    return r;
}
float BoostedTree::classify (const DataSet& data, double alpha, vector<float>& preds) const {
    float r = 0.0;
    preds.clear();
    preds.resize(data.size(), 0);
    for(int i = 0; i < data.size(); i++) {
        preds[i] += alpha * classify(data[i], alpha);
        r += (data[i]->label - preds[i]) * (data[i]->label - preds[i]);
    }
    return r/2.0;
}
