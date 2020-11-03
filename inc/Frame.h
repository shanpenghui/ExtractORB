//
// Created by sph on 2020/11/3.
//

#ifndef EXTRACTORB_INC_FRAME_H_
#define EXTRACTORB_INC_FRAME_H_
#include <vector>
#include <opencv2/opencv.hpp>
#include "Pinhole.h"

void
UndistortKeyPoints(cv::Mat &mDistCoef, std::vector<cv::KeyPoint> &mvKeys, std::vector<cv::KeyPoint> mvKeysUn, int N,
                   ORB_SLAM3::GeometricCamera *mpCamera, cv::Mat mK);

#endif //EXTRACTORB_INC_FRAME_H_
