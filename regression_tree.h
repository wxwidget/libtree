#ifndef AM_DECISION_TREE_ML_H
#define AM_DECISION_TREE_ML_H

#include <map>
#include "data_set.h"
#include "args.h"
#include "index.h"
using namespace std;
typedef enum {
    dtnLeaf,
    dtnDiscrete,
    dtnContinuous,
    dtnSubset,
    dtnGrowing
} TreeNodeType;

class RegressionTree {
public:
    enum {YES, NO, MISSING, CHILDTYPES};
    RegressionTree* child[CHILDTYPES];
    int  fid;  //feature index
    float fvalue; // feature and value this node splits on
    bool leaf;    //is the node a leaf
    float pred;  // what this node predicts if leaf node
    int datasize;

    RegressionTree(const DataSet& data, const Index& index, const args_t& myargs, int depth = 1);

    bool find_split(const DataSet& data, const Index& index, const args_t& myargs, int& fsplit, float& vsplit);

    void feature_sample(int num, int K, vector<int>& featureSelected) const;

    bool entropy_split(const DataSet& data, const Index& index, const args_t& conf, int& fsplit, float& vsplit);

    bool rmse_split(const DataSet& data, const Index& index, const args_t& conf, int& fsplit, float& vsplit);

    void sort_data_by_feature(vector<int>& location, const Index& index, const vector<int>& dataCount,
                              const vector<int>& invertIdx, int feature);
    // input: vector of Instances (data), children vector, feature and value to split on
    // output: split Instances in data into appropriate children vectors
    void split_data(const DataSet& data, DataSet child[CHILDTYPES], int f, float v);

    void print(int depth = 0);
    // destructor: free up allocated memory
    ~RegressionTree() {
        if(!leaf) {
            delete child[YES];
            delete child[NO];
            if(child[MISSING] != 0)
                delete child[MISSING];
        }
    }

    // input: vector of Instances
    // output: bool indicating if all instances in data share same label
    bool same(const DataSet& d) {
        float first = d[0]->target;
        for(int i = 1; i < d.size(); i++)
            if(d[i]->weight > 0 && d[i]->target != first)
                //if (abs(d[i]->label - first) > 0.00001)
                return false;
        return true;
    }

    // input: list of Instances
    // output: average of classes
    float average(const DataSet& data) {
        float sum = 0.0;
        float sumweight = 0.0;
        size_t n = data.size();
        for(size_t i = 0; i < n; i++) {
            sum += data[i]->target * data[i]->weight;
            sumweight += data[i]->weight;
        }
        return sum / sumweight;
    }
    float mode(const DataSet& data) {
        float best, max_count = -1;
        map<float, float> count;
        for(int i = 0; i < data.size(); i++) {
            float t = data[i]->target;
            count[t] += data[i]->weight;
            float t_count = count[t];
            if(t_count > max_count)
                max_count = t_count, best = t;
        }
        return best;
    }
    float classify(const DataSet& test, vector<float>& preds);

    float classify(const Instance* ins);
    /*
    // classify test points on tree, return rmse
    float classify_all(const DataSet& test, const RegressionTree* const tree, preds_t& preds, const args_t& conf) {
        int i, N = test.size();
        float r = 0.0;
        for(i = 0; i < N; i++) {
            preds[i] += conf.alpha * RegressionTree::classify(test[i], tree);
            r += squared(test[i]->label - preds[i]);
        }
        return sqrt(1.0 / N * r);
    }
     */
};

#endif //AM_DECISION_TREE_ML_H
