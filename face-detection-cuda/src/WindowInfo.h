#include <opencv2/core.hpp>
#include <iostream>


class WindowInfo
{
public:
    WindowInfo(cv::Mat image, int subWindowSize);
    ~WindowInfo();

    int xWindows();
    int yWindows();

    int xOffsetWindows();
    int yOffsetWindows();
    int xyOffsetWindows();

    int totalWindows();

    int subWindowSize();

    int windowOffset();
    int *subWindowOffsets();

private:
    void computePositions();

    cv::Mat mImage;
    int mSubWindowSize;
    int mOffset;
    int *mOffsets;
};