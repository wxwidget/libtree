#include <iostream>
using namespace std;

void csvformat(const char* in_string) {
    const char* pos = in_string;
    for(; (pos - 1 != NULL && pos[0] != '\0'  && pos[0] != '#');
        pos = strchr(pos, ',') + 1) {
        float value = atof(pos);
        cout << value << "\n";
    }
    return;
}
int main(int argc, char** argv){
    char* a = "1,1.2,3";
    csvformat(a);
    char* b = "1,,3";
    csvformat(b);
}
