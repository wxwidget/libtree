#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "data_set.h"
using namespace std;

void die(const std::string& reason) {
    std::cerr << "Wrong format for input data:\n  " << reason << std::endl;
    exit(1);
};

DataSet::DataSet(bool hold): holdStorage(hold), mNumFeature(0) {
}

DataSet::~DataSet() {
    if(holdStorage) {
        for(int i = 0; i < size(); ++i) {
            delete at(i);
        }
    }
}

int DataSet::read(const std::string& filename, const args_t& conf, int buffer_mb) {
    mNumFeature = conf.features;
    long int buffer_size = buffer_mb * 1024 * 1024;
    char* local_buffer = new char[buffer_size];
    std::ifstream file_stream(filename.c_str(), std::ifstream::in);
    file_stream.rdbuf()->pubsetbuf(local_buffer, buffer_size);
    if(!file_stream) {
        std::cerr << "Error reading file " << filename << std::endl;
        return 1;
    }

    string line_string;
    while(getline(file_stream, line_string)) {
        if (!check(line_string)){
            die("wrong format, should be:  label [q:1] [name:value]");
        }
        addVector(line_string, conf);
    }
    delete[] local_buffer;
    return 0;
}
bool DataSet::check(const std::string& str){
    return str.find(':') > 0;
}
DataSet::DataSet(const string& file_name, const args_t& conf, int buffer_mb): holdStorage(true), mNumFeature(conf.features) {
    read(file_name, conf, buffer_mb);
}

string DataSet::asString() const {
    string out_string;
    for(unsigned long int i = 0; i < size(); ++i) {
        out_string += at(i)->asString() + "\n";
    }
    return out_string;
}

void DataSet::addVector(const string& vector_string, const args_t& conf) {
    int num_feature = conf.features;
    float mv = std::numeric_limits<float>::quiet_NaN();
    if(0 == conf.missing) {
        mv = 0;
    }
    Instance* a = new Instance(num_feature, NULL, mv);
    if (conf.data_type == DATA_CSV_NOHEAD){
        a->init_csvformat(vector_string.c_str());
    }
    else{
        a->init_svmformat(vector_string.c_str());
    }
    push_back(a);
}

void DataSet::addIndex() {
    for(unsigned int i = 0; i < size(); ++i) {
        this->operator[](i)->idx = i;
    }
}
