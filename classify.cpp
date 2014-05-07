#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
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
    for(int i = 0; i < preds.size(); ++i){
        sum += (data[i]->label - preds[i]) *  (data[i]->label - preds[i]);
        printf("%d %f\n", (int)data[i]->label, (int)round(preds[i]), preds[i]);
    }
    fprintf(stderr, "rmse:%f\n", sum/2);
    return 0;
}
