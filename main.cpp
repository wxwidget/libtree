#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
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
        printf("cart usage: [-options] train.txt output.txt\n");
        printf("\nRequired flags:\n");
        printf("-f int\tNumber of features in the data sets. \n");
        printf("\n");
        printf("\nOptional flags:\n");
        printf("-B \tgradient boosting mode (standalone).\n");
        printf("-F \trun random forest (standalone).\n");
        printf("-a float\t stepsize.\n");
        printf("-d int \tmax treep depth (for gradient boosting trees are typically limited to a small depth, e.g. d=4).\n");
        printf("-p int\tnumber of processors/threads to use.\n");
        printf("-k \tnumber of randomly selected features used for building each trees of a random forest.\n");
        printf("-t int \tnumber of trees for random forest.\n");
        printf("-n int \tmin number of instance per node.\n");
        printf("-m \tuse mode for prediction at leaf nodes (default is mean)\n");
        printf("-z \tsubstitute missing features with zeros (recommended if missing features exist).\n");
        printf("-e \tuse entropy to measure impurity for CART (default is squared loss).\n");
        printf("-r \tnumber of boosting tree (round).\n");
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
                        fprintf(stdout, "label:%f,predict:%f\n", train[i]->label, train_preds[i]);
                    }
                }
                if(myargs.verbose) {
                    fprintf(stderr, "%d,%f", T, (float)train_rmse);
                    fprintf(stderr, "\n");
                    t->print();
                }
            }
        }

        // forest
        if(myargs.alg == ALG_FOREST) {
            //TODO
        }
    }
    return 0;
}
