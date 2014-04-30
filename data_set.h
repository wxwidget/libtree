#ifndef DATA_SET_H_
#define DATA_SET_H_
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include "args.h"
extern void die(const std::string& reason);
class Instance { // represents a data instance
public:
    float* features;
    int  numfeature;
    float label;
    float target;
    float weight;
    int qid;
    float pred;
    int idx;
    Instance(int num_features, float* init, float missingValue) : 
        numfeature(num_features), weight(1.0), label(-1), target(-1), qid(-1), pred(-1), idx(0) {
        features = new float[num_features];
        for(int i = 0; i < num_features; i++)
            if(!init)
                features[i] = missingValue;
            else
                features[i] = init[i];
    }
    ~Instance() {
        delete[] features;
    }
    std::string asString() const {
        return "";
    }
    void init_csvformat(const char* in_string) {
        if(!sscanf(in_string, "%f", &target))
            die("Class label must be real number.");
        const char* pos = strchr(in_string, ',') + 1;
        int i = 0;
        for(; pos - 1 != NULL && pos[0] != '\0' && pos[0] != '#';
                pos = strchr(pos, ',') + 1) {
            float value = atof(pos);
            if (i > numfeature)
                die("index i extend feature number");
            features[i++] = value;
        }
    }
    void init_svmformat(const char* in_string) {
        size_t length = strlen(in_string);
        if(length == 0) die("Empty example string.");
        // Get class label.
        if(!sscanf(in_string, "%f", &target))
            die("Class label must be real number.");
        // Parse the group id, if any.
        this->label = target;
        const char* position;
        position = strchr(in_string, ' ') + 1;
        if((position[0] >= 'a' && position[0] <= 'z') || (position[0] >= 'A' && position[0] <= 'Z')) {
            position = strchr(position, ':') + 1;
            qid = atoi(position);
            position = strchr(position, ' ') + 1;
        }
        // Get feature:value pairs.
        for(; (position < in_string + length
                && position - 1 != NULL
                && position[0] != '#');
                position = strchr(position, ' ') + 1) {
            // Consume multiple spaces, if needed.
            if(position[0] == ' ' || position[0] == '\n' ||
                    position[0] == '\v' || position[0] == '\r') {
                continue;
            };
            // Parse the feature-value pair.
            int id = atoi(position);//id starts with 0
            if(id >= numfeature) {
                die("feature id is larger than the no. of feature");
            }
            position = strchr(position, ':') + 1;
            float value = atof(position);
            features[id] = value;
        }
    }
};
class DataSet: public std::vector<Instance*> {
public:
    DataSet();
    DataSet(const std::string& file_name, const args_t& conf, int buffer_mb = 128);
    ~DataSet();
    // Debug string.
    std::string asString() const;
    // Number of total examples in data set.
    // Returns a reference to the specified vector.
    void addVector(const std::string& vector_string, const args_t& conf);
    int read(const std::string& filename, const args_t& conf, int buffer_mb = 128);
    void addIndex();
    int  numFeature() const {
        return mNumFeature;
    }
    bool check(const std::string& str);
private:
    DataSet(DataSet&) {};
    bool holdStorage;
    int  mNumFeature;
};
#endif
