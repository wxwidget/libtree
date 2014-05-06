#include "boosted.h"
float BoostedTree::classify(const Instance* const instance) const {
    double r = 0.0;
    const_iterator it = begin();
    for(; it != end(); ++it) {
        r = r + (*it)->weight * (*it)->classify(instance);
    }
    return r;
}
float BoostedTree::classify(const DataSet& data, vector<float>& preds) const {
    float r = 0.0;
    preds.clear();
    preds.resize(data.size(), 0);
    for(int i = 0; i < data.size(); i++) {
        preds[i] += classify(data[i]);
        r += (data[i]->label - preds[i]) * (data[i]->label - preds[i]);
    }
    return r/2.0;
}
float BoostedTree::train(const DataSet& data, const Index& index, const args_t& myargs){
    int T = 0;
    vector<float> train_preds(data.size(),0);
    vector<float> last_preds(data.size(),0);
    for(int T = 0; T < myargs.trees; T++) {
        // make tree
        RegressionTree* t = new RegressionTree(data, index, myargs,1);
        // get classification on training data
        float train_rmse = t->classify(data, train_preds);
        // update predict
        for(size_t i = 0; i < data.size(); i++) {
            if (0 == T){
                last_preds[i] += train_preds[i];
                t->weight = 1;
            }else{
                last_preds[i] += myargs.alpha * train_preds[i];
                t->weight = myargs.alpha;
            }
        }
        // update targets
        for(size_t i = 0; i < data.size(); i++) {
            data[i]->target = data[i]->label - last_preds[i];
        }
        if(myargs.verbose) {
            fprintf(stderr, "\n");
            t->print();
        }
        push_back(t);
    }
}
BoostedTree::~BoostedTree(){
    iterator it = begin();
    for(; it != end();++it){
        delete *it;
    }
}
