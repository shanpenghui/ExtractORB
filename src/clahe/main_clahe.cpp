//
// Created by sph on 2020/11/5.
//
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    cv::Mat im;
    im = cv::imread("/home/sph/Documents/orbslam/ExtractORB/pic/robot/866_im.jpg",cv::IMREAD_GRAYSCALE);
    cv::Mat im_clahe;
    clahe->apply(im, im_clahe);

    cv::imshow("src image", im);
    cv::imshow("dst image", im);
    cv::waitKey(0);
    return 0;
}