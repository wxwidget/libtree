#include <cassert>
#include <iostream>
#include "regression_tree.h"
#include "args.h"
using namespace std;
int main(int argc, char** argv){
    args_t conf;
    conf.features = 2;
    conf.minInstance = 1;
    conf.kfeatures = 2;
    conf.data_type = DATA_CSV_NOHEAD;
    RegressionTree node;
    DataSet data;
    data.setNumFeature(conf.features);
    data.addVector("0,1,0,0",conf);
    data.addVector("0,1,0,1",conf);
    data.addVector("0,3,0,0",conf);
    data.addVector("1,7,1,1",conf);
    data.addVector("1,5,0,0",conf);
    data.addVector("1,8,1,1",conf);
    data.addVector("1,9,1,1",conf);
    conf.ntra = data.size();
    data.addIndex();
    Index index(data);
    int fsplit;
    float vsplit;
    node.rmse_split(data, index, conf, fsplit, vsplit);
    cout << fsplit << " " << vsplit << endl;;
    assert(fsplit==0);
    assert(vsplit==4);
    return 0;
}
