#ifndef CPP_EXAMPLE_DISPLAY2D_H
#define CPP_EXAMPLE_DISPLAY2D_H

#include "imaging/DataType.h"

class Display2D {
public:
    Display2D() = default;

    void update(void *inputPtr) {
        if(isClosed) {
            windowClosed.notify_all();
            return;
        }
        cv::Mat mat(nrows, ncols, inputDataTypeCv, inputPtr);
        if(inputDataTypeCv == CV_32FC2) {
            cv::Mat real(nrows, ncols, CV_32F);
            cv::Mat imag(nrows, ncols, CV_32F);
            cv::Mat abs(nrows, ncols, CV_32F);
            cv::Mat logAbs(nrows, ncols, CV_32F);
            cv::Mat logAbsColor;
            Mat out[] = {real, imag};
            int from_to[] = {0,0 , 1,1};
            cv::mixChannels(&mat, 1, out, 2, from_to, 2);
            cv::Mat mag = real.mul(real) + imag.mul(imag);
            ::cv::imshow("Display2D", real);
        }
        else {
            ::cv::transpose(mat, mat);
            ::cv::imshow("Display2D", mat);
        }
        // Refresh the window and check if user pressed 'q'.
        int key = waitKey(1);
    }

    void exit() {
        windowClosed.notify_all();
    }

    void close() {
        isClosed = true;
    }

    void waitUntilClosed(std::unique_lock<std::mutex> &lock) {
        windowClosed.wait(lock);
    }

    void setNrows(unsigned int nrows) {
        Display2D::nrows = nrows;
    }

    void setNcols(unsigned int ncols) {
        Display2D::ncols = ncols;
    }

    void setInputDataType(imaging::DataType inputDataType) {
        switch(inputDataType) {
            case imaging::DataType::INT16:
                inputDataTypeCv = CV_16S;
                break;
            case imaging::DataType::UINT8:
                inputDataTypeCv = CV_8U;
                break;
            case imaging::DataType::FLOAT32:
                inputDataTypeCv = CV_32F;
                break;
            case imaging::DataType::COMPLEX64:
                inputDataTypeCv = CV_32FC2;
                break;
            default:
                throw std::runtime_error("Invalid input data type.");
        }
    }

private:
    unsigned int nrows, ncols;
    bool isClosed{false};
    size_t inputDataTypeCv;
    std::condition_variable windowClosed;
};

#endif //CPP_EXAMPLE_DISPLAY2D_H
