#include "gbdt.h"
#include "serialize.h"
#include <cmath>
float GBDTree::classify (const Instance* const instance) const {
    double r = 0;
    const_iterator it = begin();
    for(; it != end(); ++it) {
        r = r + (*it)->weight * (*it)->classify(instance);
    }
    return r;
}

float GBDTree::train(const DataSet& data, const Index& index, const args_t& myargs){
    int sz = data.size();
    float sum = 0;
    for(int i = 0; i < data.size(); ++i){
        sum += data[i]->label;
    }
    double T0 = 0;
    /*
    RegressionTree* t0 = new RegressionTree();
    T0 = t0->pred = sum/sz;
    push_back(t0);
    for(int i = 0; i < data.size(); ++i){
        float target = data[i]->target;
        data[i]->target = target - T0;
    }
    */
    vector<float> preds(data.size(),0);
    vector<float> gradient(data.size(),0);
    vector<float> lastPredict(data.size(),T0);
    for(int i = 0; i < myargs.rounds; i++){
        RegressionTree* tree = new RegressionTree(data, index, myargs); 
        tree->classify(data, preds);
        if(0 && myargs.verbose) {
            for(int i = 0; i < preds.size(); ++i){
                fprintf(stderr,"label:%f,target:%f,predict:%f\n", data[i]->label,data[i]->target,preds[i]);
            }
            //t->print();
        }
        //gradient(data, preds);
        for(int i = 0; i < data.size(); ++i){
            float target = data[i]->target;
            gradient[i] = target - preds[i];
        }
        // currentPredict = lastPredict + alpha * gradient
        // find the best alpha
        // min Loss(lastPredict + alpha * gradient, label)
        // lastPredict + alpha * gradient = label
        // alpha * gradient = label - lastPredict
        // Ax = y
        // x = A^T(A^T*A)y
        double ata = 0.0;
        double aty = 0;
        for (int i = 0; i < gradient.size(); ++i){
            ata += gradient[i] * gradient[i];
            aty += gradient[i] * data[i]->label;
        }
        //tree->weight = myargs.alpha;
        double weight = (aty+1)/(1+ata);
        tree->weight = weight/sqrt(1 + weight*weight);
        double rmse = 0;
        for (int i = 0; i < lastPredict.size(); ++i){
            lastPredict[i] = lastPredict[i] + tree->weight * preds[i];
            data[i]->target = data[i]->label - lastPredict[i];
            rmse += data[i]->target * data[i]->target;
        }
        if(myargs.verbose) {
            fprintf(stderr, "tweight:%f ", tree->weight);
            fprintf(stderr, "rmse:%f\n", (float)rmse/2);
            //t->print();
        }
        push_back(tree);
    }
}
float GBDTree::classify (const DataSet& data, vector<float>& preds) const {
    float r = 0.0;
    preds.clear();
    preds.resize(data.size(),0);
    for(int i = 0; i < data.size(); i++) {
        preds[i] = classify(data[i]);
        r += (data[i]->label - preds[i]) * (data[i]->label - preds[i]);
    }
    return r/2.0;
}
void GBDTree::save(std::ostream& out){
    Serialize ser(out);
    ser << (int)this->size();
    for(int i = 0;i < this->size(); ++i){
        at(i)->save(ser);
    }
}
void GBDTree::load(std::istream&in){
    Deserialize ser(in);
    int numTrees;
    ser >> numTrees;
    for(int i = 0;i < numTrees; ++i){
        RegressionTree* tree = new RegressionTree();
        tree->load(ser);
        push_back(tree);
    }
}
GBDTree::~GBDTree(){
    iterator it = begin();
    for(; it != end();++it){
        delete *it;
    }
}
