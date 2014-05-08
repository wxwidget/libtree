#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <map>
#include "data_set.h"
#include "regression_tree.h"
#include "epoch.h"
#include "boosted.h"
using namespace std;
int main(int argc, char* argv[]) {
    if (argc != 4){
        printf("usage:classify model_file test_file numfeatures\n");
        return 0;
    }
    char* model_file = argv[1];
    ifstream model(model_file);
    char* testfile = argv[2];
    int numfeature = atoi(argv[3]);
    args_t myargs;
    myargs.features = numfeature;
    DataSet data;
    if(data.read(testfile, myargs) != 0) {
        die("could not load data files\n");
    }
    BoostedTree trees;
    trees.load(model);
    vector<float> preds;
    trees.classify(data, preds);
    double sum = 0.0;
    int correct = 0;
    typedef map<int, int> Row;
    map<int, Row> matrix;
    for(int i = 0; i < preds.size(); ++i){
        sum += (data[i]->label - preds[i]) *  (data[i]->label - preds[i]);
        int label = (int)data[i]->label;
        int pred = (int)round(preds[i]);
        if (label==pred){
            correct++;
        }
        matrix[label][pred] += 1;
        printf("%d %d\n", (int)data[i]->label, (int)round(preds[i]));
    }
    fprintf(stderr, "rmse:%f\n", sum/2);
    fprintf(stderr, "correct instance :%d of %d, pre:%f\n", correct, data.size(),float(correct)/data.size());
    map<int, Row>::const_iterator it = matrix.begin();
    for(; it != matrix.end(); ++it){
        int label = it->first;
        const Row& predic = it->second;
        fprintf(stderr,"label:%d\t", label);
        Row::const_iterator rit = predic.begin();
        for(; rit != predic.end(); ++rit){
            fprintf(stderr,"%d:%d\t", rit->first, rit->second);
        }
        fprintf(stderr,"\n");
    }
    return 0;
}
