#include <stdint.h>
#include "serialize.h"

Serialize::Serialize(std::ostream& s)
    : mpStream(&s)
{
}

Serialize::~Serialize()
{
}

Serialize& Serialize::operator << (const std::string& strData)
{
    size_t nLen = strData.size();
    (*this) << nLen;
    if (nLen > 0)
    {
        mpStream->write(strData.c_str(), nLen);
    }
    return *this;
}

Serialize& Serialize::operator << (const size_t& data)
{
    uint64_t nTemp = static_cast<uint64_t>(data);
    mpStream->write(reinterpret_cast<const char*>(&nTemp), sizeof(uint64_t));
    return *this;
}

bool Serialize::bad()
{
    return mpStream->bad();
}

Deserialize::Deserialize(std::istream& s)
    :mpStream(&s)
{
}

Deserialize::~Deserialize()
{
}

Deserialize& Deserialize::operator >> (std::string& strData)
{
    size_t nLen = 0;
    (*this) >> nLen;
    if (nLen > 0)
    {
        char cBuf[nLen + 1];
        mpStream->read(cBuf, nLen);
        cBuf[nLen] = '\0';
        strData = cBuf;
    }
    return *this;
}

Deserialize& Deserialize::operator >> (size_t& data)
{
    uint64_t nTemp;
    mpStream->read(reinterpret_cast<char*>(&nTemp), sizeof(uint64_t));
    data = static_cast<size_t>(nTemp);
    return *this;
}

bool Deserialize::bad()
{
    return mpStream->bad();
}

