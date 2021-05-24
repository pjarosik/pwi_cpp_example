#ifndef RECONSTRUCTION_RATE_TEST__COMMON_H_
#define RECONSTRUCTION_RATE_TEST__COMMON_H_

#include <fstream>

std::vector<float> arange(float l, float r, float step) {
    std::vector<float> result;
    float currentPos = l;
    while(currentPos <= r) {
        result.push_back(currentPos);
        currentPos += step;
    }
    return result;
}

std::vector<float> linspace(float l, float r, unsigned n) {
    std::vector<float> result(n, 0.0f);
    float step = (r-l)/(float)(n-1);
    float currentAngle = l;
    for(int i = 0; i < n; ++i) {
        result[i] = currentAngle;
        currentAngle += step;
    }
    return result;
}

void writeDataToFile(const std::string& path, char* dataPtr, size_t nBytes) {
    std::cerr << "NOTE: writing data to file - it make impact processing performance" << std::endl;
    std::ofstream file;
    file.open(path, std::ios_base::binary);
    file.write((char*)dataPtr, nBytes);
    file.close();
}


#endif //RECONSTRUCTION_RATE_TEST__COMMON_H_
