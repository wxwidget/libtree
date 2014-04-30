#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include "data_set.h"
#include "index.h"
#include "regression_tree.h"
#include "args.h"
#include "epoch.h"
using namespace std;
int main(int argc, char* argv[]) {
    srand(time(NULL));
    // get command line args
    args_t myargs;
    if(myargs.get_args(argc, argv)) {
        printf("tree usage: [-options] train.txt \n");
        printf("\t options=[f:a:d:l:i:k:t:o:m:p:r:n:h:vzBFR]");
        printf("\nRequired flags:\n");
        printf("-f int\tNumber of features in the data sets. \n");
        printf("\n");
        printf("\nOptional flags:\n");
        printf("-a \tfloat, stepsize.[default 1] \n");
        printf("-d \tint, max treep depth [default 4](for gradient boosting trees are typically limited to a small depth, e.g. d=4).\n");
        printf("-l \tstring=[squared, entropy],loss functions [default is squared loss].\n");
        printf("-p \tint,  number of processors/threads to use [default 1] .\n");
        printf("-o \tstring, output the tree\n");
        printf("-k \tint,number of randomly selected features used for building each trees of a random forest [default features]\n");
        printf("-t \tint, number of trees for random forest.[default 1]\n");
        printf("-n \tint, min number of instance per node.[default 10] \n");
        printf("-m \tstring=[mode,mean] use mode for prediction at leaf nodes.[default mean] \n");
        printf("-r \tnumber tress of gbdt.[default 1 ]\n");
        printf("-v \tprint debuging info .\n");
        printf("-z \tsubstitute missing features with zeros (recommended if missing features exist).\n");
        printf("-h \tstring[csv,svm], data format, csv: with no head, svm : libsvm or ranksvm");
        printf("-B \tgradient boosting mode (standalone).\n");
        printf("-F \trun random forest (standalone).\n");
        printf("-R \trun regression tree (standalone).\n");
        printf("\n\n");
        return 0;
    }
    if(myargs.kfeatures < 1) {
        myargs.kfeatures = myargs.features;
    }
    // load data from input files
    DataSet train;
    if(train.read(myargs.train_file, myargs) != 0) {
        die("could not load data files\n");
    }
    train.addIndex();
    // presort the training data for each feature
    Index index(train);
    vector<RegressionTree*> trees;
    myargs.ntra = train.size();
    vector<float> train_preds(train.size(), 0);
    for(int round = 0; round < myargs.rounds; round++) {
        size_t N = train.size();
        vector<int> countData(myargs.ntra, 1);
        if(myargs.alg == ALG_BOOST || myargs.alg == ALG_REGRESSION) {
            for(int T = 0; T < myargs.trees; T++) {
                // make tree
                RegressionTree* t = new RegressionTree(train, index, myargs);
                // get classification on training data
                float train_rmse = t->classify(train, train_preds);
                // update targets
                for(size_t i = 0; i < N; i++) {
                    train[i]->target = train[i]->label - train_preds[i];
                    if(myargs.verbose) {
                        //fprintf(stdout, "label:%f,predict:%f\n", train[i]->label, train_preds[i]);
                    }
                }
                if(myargs.verbose) {
                    fprintf(stderr, "Tree:%d,mse:%f", T, (float)train_rmse);
                    fprintf(stderr, "\n");
                    t->print();
                }
                trees.push_back(t);
            }
        }
        // forest
        if(myargs.alg == ALG_FOREST) {
            //TODO
        }
    }
    if (myargs.output_file){
        ofstream of(myargs.output_file,std::ofstream::out);
        for(int i = 0; i < trees.size(); ++i){
            trees[i]->save(of);
            delete trees[i];
        }
    }
    return 0;
}
