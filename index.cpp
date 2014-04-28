#include <cmath>
#include <algorithm>
#include "index.h"

using namespace std;
bool mysortpred2(const pair<float, int> tk1, const pair<float, int > tk2) {
    return tk1.first < tk2.first;
}
Index::Index(const DataSet& dt) {
    int numfeatures = dt.numFeature();
    this->resize(numfeatures);
    // sort data
    for(int f = 0; f < numfeatures; f ++) {
        IntVector index;
        typedef pair<float, int> FeatureType;
        vector<FeatureType> tk;
        for(int z = 0; z < dt.size(); z++) {
            float value = dt[z]->features[f];
            int id = dt[z]->idx;
            if(!isnan(value)) {
                tk.push_back(pair<float, int >(value, id));
            }
        }
        sort(tk.begin(), tk.end(), mysortpred2);
        index.reserve(tk.size());
        for(int z = 0; z < tk.size(); z++) {
            index.push_back(tk[z].second);
        }
        at(f).swap(index);
    }
};
