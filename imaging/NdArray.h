#ifndef CPP_EXAMPLE_NDARRAY_H
#define CPP_EXAMPLE_NDARRAY_H

#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

#include "CudaUtils.cuh"
#include "DataType.h"

namespace imaging {
/**
 * This class owns memory
 */

class NdArray {
public:
    typedef std::vector<unsigned> DataShape;
    typedef DataType DataType;

    NdArray() {}

    NdArray(const NdArray::DataShape &shape, NdArray::DataType dataType,
            bool isGpu) : ptr(nullptr), shape(shape), dataType(dataType),
                          isGpu(isGpu) {
        if (shape.empty()) {
            // empty array shape (0)
            return;
        }
        size_t result = 1;
        for (auto &val: shape) {
            result *= val;
        }
        nBytes = result * getSizeofDataType(dataType);
        if (isGpu) {
            CUDA_ASSERT(cudaMalloc(&ptr, nBytes));
        } else {
            CUDA_ASSERT(cudaMallocHost(&ptr, nBytes));
        }
    }

    NdArray(void *ptr, const DataShape &shape, DataType dataType, bool isGpu)
            : ptr((char *) ptr), shape(shape), dataType(dataType),
              isGpu(isGpu) {
        size_t flattenShape = 1;
        for (auto &val: shape) {
            flattenShape *= val;
        }
        nBytes = flattenShape*getSizeofDataType(dataType);
        isExternal = true;
    }

    NdArray(NdArray &&array) noexcept: ptr(array.ptr),
                                       shape(std::move(array.shape)),
                                       dataType(array.dataType),
                                       isGpu(array.isGpu),
                                       nBytes(array.nBytes) {
        array.ptr = nullptr;
        array.nBytes = 0;
    }

    NdArray &operator=(NdArray &&array) noexcept {
        if (this != &array) {
            freeMemory();

            ptr = array.ptr;
            array.ptr = nullptr;

            nBytes = array.nBytes;
            array.nBytes = 0;

            shape = std::move(array.shape);
            dataType = array.dataType;
            isGpu = array.isGpu;
        }
        return *this;
    }

    NdArray(const NdArray &) = delete;

    NdArray &operator=(NdArray const &) = delete;

    virtual ~NdArray() {
        freeMemory();
    }

    template<typename T>
    T *getPtr() {
        return (T *) ptr;
    }

    template<typename T>
    const T *getConstPtr() const {
        return (T *) ptr;
    }

    const std::vector<unsigned> &getShape() const {
        return shape;
    }

    DataType getDataType() const {
        return dataType;
    }

    size_t getNBytes() const {
        return nBytes;
    }

    bool IsGpu() const {
        return isGpu;
    }


    void freeMemory() {
        if (ptr == nullptr) {
            return;
        }
        if (isGpu) {
            CUDA_ASSERT_NO_THROW(cudaFree(ptr));
        } else if (!isExternal) {
            CUDA_ASSERT_NO_THROW(cudaFreeHost(ptr));
        }
        // external data is not managed by this class
    }

    NdArray copyToDevice() const {
        NdArray result(this->shape, this->dataType, true);
        CUDA_ASSERT(cudaMemcpy(result.ptr, this->ptr, this->nBytes,
                               cudaMemcpyHostToDevice));
        return result;
    }

    NdArray copyToHost() const {
        NdArray result(this->shape, this->dataType, false);
        CUDA_ASSERT(cudaMemcpy(result.ptr, this->ptr, this->nBytes,
                               cudaMemcpyDeviceToHost));
        return result;
    }

    NdArray copyToHostAsync(cudaStream_t const &stream) const {
        NdArray result(this->shape, this->dataType, false);
        CUDA_ASSERT(cudaMemcpyAsync(result.ptr, this->ptr, this->nBytes,
                                    cudaMemcpyDeviceToHost, stream));
        return result;
    }


private:
    static size_t getSizeofDataType(DataType type) {
        if (type == DataType::INT16) {
            return sizeof(short);
        } else if (type == DataType::UINT16) {
            return sizeof(unsigned short);
        } else if (type == DataType::FLOAT32) {
            return sizeof(float);
        } else if (type == DataType::COMPLEX64) {
            return sizeof(float) * 2;
        } else if (type == DataType::UINT8) {
            return sizeof(unsigned char);
        } else if (type == DataType::INT8) {
            return sizeof(char);
        }
        throw std::runtime_error("Unhandled data type");
    }

    char *ptr{nullptr};
    DataShape shape;
    size_t nBytes;
    DataType dataType;
    bool isGpu;
    bool isExternal{false};
};
}

#endif //CPP_EXAMPLE_NDARRAY_H
