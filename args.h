//parse command line arguments
#ifndef AM_RT_ARGS_H
#define AM_RT_ARGS_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
using namespace std;

enum AlgType { ALG_BOOST, ALG_GBDT, ALG_FOREST, ALG_REGRESSION };
enum LossType { ALG_SQUARED_LOSS, ALG_ENTROPY };
enum PredictType { ALG_MEAN, ALG_MODE };
enum DataType{DATA_SVM, DATA_CSV_NOHEAD};
struct args_t {
    int features; //number of features
    int trees;    //num of trees
    int processors;//for multiprocessor
    double alpha; //learning rate
    int depth;    //max depth of the tree
    int kfeatures;//random k features
    int minInstance;// mininum number of instance in the leaf to support the value;
    char* train_file;
    char* output_file;
    int verbose;  //
    int rounds;
    AlgType alg;
    LossType loss;
    PredictType pred;
    DataType  data_type;
    int missing;// 1, using nan as missing value, 0 use default value for missing value
    char* missing_file;//if missing == 0,
    unsigned long ntra;   //number of training data
    args_t() {
        processors = 1;
        trees = 1;
        alpha = 1.0;
        features = 701;
        depth = 4;
        kfeatures = -1;
        minInstance = 10;
        train_file = NULL;
        output_file = NULL;
        alg = ALG_REGRESSION;
        verbose = 0;
        rounds = 1;
        loss = ALG_SQUARED_LOSS;
        pred = ALG_MEAN;
        data_type = DATA_SVM;
        missing = 1;
        missing_file = NULL;
    };
    LossType  lossMapping(const char* str){
        if (strcmp(str, "squared") == 0){
            return ALG_SQUARED_LOSS;
        }else if(strcmp(str, "entropy") == 0){
            return ALG_ENTROPY;
        }
        return ALG_SQUARED_LOSS;
    }
    PredictType modeMapping(const char* str){
        if (strcmp(str, "mean") == 0){
            return ALG_MEAN;
        }else if(strcmp(str, "mode") == 0){
            return ALG_MODE;
        }
        return ALG_MEAN;
    }
    DataType formatMapping(const char* str){
        if (strcmp(str, "csv") == 0){
            return DATA_CSV_NOHEAD;
        }
        return DATA_SVM;
    }
    int get_args(int argc, char** argv) {
        int c, i = 0;
        // option arguments
        opterr = 0;
        while((c = getopt(argc, argv, "f:a:d:l:t:i:k:o:m:p:r:n:t:vzGBFR")) != -1)
            switch(c) {
            case 'a':
                this->alpha = atof(optarg);
                break;
            case 'd':
                this->depth = atoi(optarg);
                break;
            case 'l':
                this->loss = lossMapping(optarg);
                break;
            case 'm':
                this->pred = modeMapping(optarg);
                break;
            case 't':
                this->trees = atoi(optarg);
                break;
            case 'o':
                this->output_file = optarg;
                break;
            case 'p':
                this->processors = atoi(optarg);
                break;
            case 'f':
                this->features = atoi(optarg);
                break;
            case 'k':
                this->kfeatures = atoi(optarg);
                break;
            case 'r':
                this->rounds = atoi(optarg);
                break;
            case 'n':
                this->minInstance = atoi(optarg);
                break;
            case 'h':
                this->data_type = formatMapping(optarg);
                break;
            case 'i':
                this->missing_file = optarg;
                break;
            case 'B':
                this->alg = ALG_BOOST;
                break;
            case 'F':
                this->alg = ALG_FOREST;
                break;
            case 'R':
                this->alg = ALG_REGRESSION;
                break;
            case 'G':
                this->alg = ALG_GBDT;
                break;
            case 'v':
                this->verbose = 1;
                break;
            case 'z':
                this->missing = 0;
                break;
            case '?':
                if(optopt == 'c')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if(isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                return 1;
            default:
                return 1;
            }
        // non option arguments
        if(argc - optind < 1) {
            fprintf(stderr, "no option arguments argc:%d, optind%d\n", argc, optind);
            return 1;
        }
        if(i == 0) this->train_file = argv[optind];
        return 0;
    }
};
#endif
