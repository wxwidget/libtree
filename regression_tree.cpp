#include <algorithm>
#include <cmath>
#include "regression_tree.h"
#include "serialize.h"
using namespace std;

RegressionTree::RegressionTree():fid(-1), fvalue(-1), leaf(true), pred(-1), datasize(-1), weight(1.0f){
    child[YES] = NULL;
    child[NO] = NULL;
    child[MISSING] = NULL;
}
RegressionTree::RegressionTree(const DataSet& data, const Index& index, const args_t& conf, int depth):
    leaf(false), fid(-1), fvalue(-1), pred(-1), datasize(data.size()),weight(1.0f) {
    for(int i = 0; i < CHILDTYPES; i++) {
        child[i] = NULL;
    }
    // get prediction
    pred = (conf.pred == ALG_MEAN) ? average(data) : mode(data);

    // check if leaf node
    if(depth == conf.depth || data.size() <= conf.minInstance || same(data)) {
        leaf = true;
        return;
    }

    // find criterion to split data, if cant make this leaf node
    int f_split;
    float v_split;
    if(!find_split(data, index, conf, f_split, v_split)) {
        leaf = true;
        return;
    }
    // split data into 3 parts, based on criteria found
    DataSet child_data[CHILDTYPES];
    split_data(data, child_data, f_split, v_split);

    if(child_data[YES].empty() || child_data[NO].empty()) {
        leaf = true;
        return;
    }
    // remember where we splitted, and recurse
    this->fid = f_split;
    this->fvalue = v_split;
    child[YES] = new RegressionTree(child_data[YES], index, conf, depth + 1);
    child[NO] = new RegressionTree(child_data[NO], index, conf, depth + 1);

    if(!child_data[MISSING].empty())
        child[MISSING] = new RegressionTree(child_data[MISSING], index, conf, depth + 1);
    else
        child[MISSING] = 0;
}
void RegressionTree::sort_data_by_feature(vector<int>& location, const Index& index,
        const vector<int>& dataCount, const vector<int>& invertIdx, int f) {
    location.clear();
    const vector<int>& instances = index[f];//sorted global feature index
    for(int i = 0; i < instances.size(); i++) {
        int z =  instances[i]; //get feature id
        int loc = invertIdx[z];//id to local location
        for(int j = 0; j < dataCount[z]; j ++)
            location.push_back(loc);
    }
}

///////////////////////
bool RegressionTree::entropy_split(const DataSet& data, const Index& dataCount, const args_t& conf, int& f_split, float& v_split) {
    return false;
}
//////////////////////
void RegressionTree::feature_sample(int num, int K, vector<int>& selected) const {
    // pick K random features to split on, if specified
    if(K >= num || K < 0) return;
    int left = K;
    int begin = 0;
    while(K-- > 0) {
        int r = begin + rand() % left;
        std::swap(selected[begin], selected[r]);
        ++begin;
        --left;
    }
}

void RegressionTree::print(int depth) {
    for(int i = 0; i < depth; ++i) {
        printf("   ");
    }
    printf("id:%d,value:%f, pred:%f,data_size:%d\n", fid, fvalue, pred, datasize);
    if(!this->leaf) {
        if(child[YES])
            child[YES]->print(depth + 1);
        if(child[NO])
            child[NO]->print(depth + 1);
        if(child[MISSING])
            child[MISSING]->print(depth + 1);
    }
}

void RegressionTree::save(Serialize& ser){
    ser << fid << fvalue << pred << datasize << weight << this->leaf;
    if(!this->leaf) {
        ser << int(child[YES] != NULL);
        ser << int(child[NO] != NULL);
        ser << int(child[MISSING] != NULL);
        if(child[YES])
            child[YES]->save(ser);
        if(child[NO])
            child[NO]->save(ser);
        if(child[MISSING])
            child[MISSING]->save(ser);
    }
}
void RegressionTree::save(std::ostream& out){
    Serialize ser(out);
    save(ser);
}

void RegressionTree::load(Deserialize& ser){
    ser >> fid >> fvalue >> pred >> datasize >> weight >> leaf;
    if(!this->leaf) {
        int yes, no, missing;
        ser >> yes >> no >> missing;
        if(yes){
            child[YES] = new RegressionTree();
            child[YES]->load(ser);
        }
        if(no){
            child[NO] = new RegressionTree();
            child[NO]->load(ser);
        }
        if(missing){
            child[MISSING] = new RegressionTree();
            child[MISSING]->load(ser);
        }
    }
}
void RegressionTree::load(std::istream& in){
    Deserialize ser(in);
    load(ser);
}
float RegressionTree::classify(const DataSet& test, vector<float>& preds) const{
    float mse = 0.0;
    for(int i = 0; i < test.size(); ++i) {
        float p = classify(test[i]);
        mse += (test[i]->target - p) * (test[i]->target - p) / 2;
        preds[i] = p;
    }
    return mse;
}
float RegressionTree::classify(const Instance* ins) const {
    float* features = ins->features;
    float  featurevalue = ins->features[this->fid];
    if(this->leaf) {
        return pred;
    }
    if(isnan(featurevalue)) {
        return child[MISSING]->classify(ins);
    }
    if(featurevalue > this->fvalue) {
        return child[NO]->classify(ins);
    }
    return child[YES]->classify(ins);
}
bool RegressionTree::rmse_split(const DataSet& data, const Index& index, const args_t& conf, int& fsplit, float& vsplit) {
    //build local index for local input dataset
    vector<int> invertIdx(conf.ntra, -1);
    vector<int> countData(conf.ntra, 0);
    for(int i = 0; i < data.size(); i ++) {
        int idx = data[i]->idx;
        invertIdx[idx] = i;    //instance id to data index
        countData[idx]++;      //instance id counts
    }
    fsplit = -1;
    float min = numeric_limits<float>::max();
    vector<int> selectedFeature;
    for(int i = 0; i < conf.features; ++i) {
        if(index[i].size() >= conf.minInstance)
            selectedFeature.push_back(i);
    }
    feature_sample(conf.features, conf.kfeatures, selectedFeature);
    size_t n = data.size();
    vector<int> location;
    location.reserve(n);
    int fcount = std::min((int)conf.kfeatures, (int)selectedFeature.size());
    for(int id = 0; id < fcount; ++id) {
        int f = selectedFeature[id];
        // sort data
        sort_data_by_feature(location, index, countData, invertIdx, f);

        float s, r, L, R, I, ywl, ywr, WL, WR;
        s = r = ywl = ywr = WL = WR = 0.0;
        // put all data into right side R
        // get ybar right and r
        int d_sz = location.size();
        for(int i = 0; i < d_sz; i++) {
            int loc = location[i];
            r += data[loc]->target * data[loc]->target * data[loc]->weight; //r = sum_i{ y_i * y_i}
            ywr += data[loc]->target * data[loc]->weight;  //ywr = sum_i{y_i}
            WR += data[loc]->weight;
        }
        //sum(y-mean_y)^2  = D(y) = E(x*x) - E(x)*E(x)
        for(int i = 0; i < d_sz - conf.minInstance; i++) {
            float yn = data[location[i]]->target;
            float w = data[location[i]]->weight;
            s += w *  yn * yn;
            r -= w * yn * yn;
            ywr -= w * yn;
            ywl += w * yn;
            WL += w;
            WR -= w;
            L = s - ywl * ywl / WL;
            R = r - ywr * ywr / WR;
            // precision errors
            if(L < 0) L = 0;
            if(R < 0) R = 0;
            // do not consider splitting here if data is the same as next
            if(data[location[i]]->features[f] == data[location[i + 1]]->features[f])
                continue;
            //I = L + R + M ;
            I = L + R;
            if(I < min && i > conf.minInstance) {
                min = I;
                fsplit = f;
                vsplit = (data[location[i]]->features[f] + data[location[i + 1]]->features[f]) / 2;
            }
        }
    }
    return min != numeric_limits<float>::max();
}

bool RegressionTree::find_split(const DataSet& data, const Index& index, const args_t& conf, int& f_split, float& v_split) {
    if(conf.loss == ALG_ENTROPY)
        return entropy_split(data, index, conf, f_split, v_split);
    else
        return rmse_split(data, index, conf, f_split, v_split);
    return false;
}

void RegressionTree::split_data(const DataSet& data, DataSet child[CHILDTYPES], int f, float v) {
    size_t n = data.size();
    for(int i = 0; i < CHILDTYPES; i++) {
        child[i].clear();
    }
    for(int i = 0; i < n; i++) {
        if(isnan(data[i]->features[f]))
            child[MISSING].push_back(data[i]);
        else if(data[i]->features[f] <= v)
            child[YES].push_back(data[i]);
        else
            child[NO].push_back(data[i]);
    }
}
