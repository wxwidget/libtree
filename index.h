#ifndef INDEX_H_
#define INDEX_H_
#include <string>
#include <vector>
#include "data_set.h"
typedef std::vector<int> IntVector;
class Index: public std::vector<IntVector> {
public:
    Index(const DataSet&);
    ~Index() {}
private:
    Index(const Index&) {};
    Index& operator=(const Index&);
};
#endif
