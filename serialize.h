#ifndef SERIALIZE_H
#define SERIALIZE_H
#include <iostream>
#include <vector>
#include <string>
class Serialize
{
public:
    Serialize(std::ostream& s);
    ~Serialize();
    template <typename Type>
    Serialize& operator << (const Type& data);//for pod types
    Serialize& operator << (const std::string& strData);
    Serialize& operator << (const size_t& data);
    template <typename Type>
    Serialize& operator << (const std::vector<Type>& vecType);
    template <typename InputIterator>
    void pack(InputIterator iterBegin, InputIterator iterEnd);
    bool bad();
private:
    std::ostream* mpStream;
};

class Deserialize
{
public:
    Deserialize(std::istream& s);
    ~Deserialize();
    template <typename Type>
    Deserialize& operator >> (Type& data);
    Deserialize& operator >> (std::string& strData);
    Deserialize& operator >> (size_t& data);
    template <typename Type>
    Deserialize& operator >> (std::vector<Type>& vecType);
    template <typename OutputIterator>
    void unpack(OutputIterator iterBegin, OutputIterator iterEnd);
    bool bad();

private:
    std::istream* mpStream;
}; // Deserialize

template <typename Type>
Serialize& Serialize::operator << (const Type& data)
{
    mpStream->write(reinterpret_cast<const char*>(&data), sizeof(Type));
    return *this;
}

template <typename Type>
Serialize& Serialize::operator << (const std::vector<Type>& vecType)
{
    size_t nSize = vecType.size();
    *this << nSize;
    for(typename std::vector<Type>::const_iterator iterType = vecType.begin();
            iterType != vecType.end(); ++iterType)
    {
        *this << *iterType;
    }
    return *this;
}

template <typename InputIterator>
void Serialize::pack(InputIterator iterBegin, InputIterator iterEnd)
{
    for(InputIterator iter = iterBegin; iter != iterEnd; ++iter)
        *this << *iter;
}

template <typename Type>
Deserialize& Deserialize::operator >> (Type& data)
{
    mpStream->read(reinterpret_cast<char*>(&data), sizeof(Type));
    return *this;
}

template <typename Type>
Deserialize& Deserialize::operator >> (std::vector<Type>& vecType)
{
    size_t nSize = vecType.size();
    *this >> nSize;
    vecType.resize(nSize);
    for(typename std::vector<Type>::iterator iterType = vecType.begin();
            iterType != vecType.end(); ++iterType)
    {
        *this >> *iterType;
    }
    return *this;
}

template <typename OutputIterator>
void Deserialize::unpack(OutputIterator iterBegin, OutputIterator iterEnd)
{
    for(OutputIterator iter = iterBegin; iter != iterEnd; ++iter)
        *this >> *iter;
}
#endif 

