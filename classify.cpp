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
    vector<float> preds;
    while(model){
        RegressionTree* t = new RegressionTree();
        t->load(model);
        trees.push_back(t);
    }
    trees.classify(data, preds);
    for(int i = 0; i < preds.size(); ++i){
        printf("%f %f\n", data[i]->target, preds[i]);
    }
    return 0;
}
