//
// Created by sph on 2020/11/3.
//

#ifndef EXTRACTORB_INC_FRAME_CHANGED_H_
#define EXTRACTORB_INC_FRAME_CHANGED_H_
#include <vector>
#include <opencv2/opencv.hpp>
#include "Pinhole.h"
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

void
UndistortKeyPoints(cv::Mat &mDistCoef, std::vector<cv::KeyPoint> &mvKeys, std::vector<cv::KeyPoint> mvKeysUn, int N,
                   ORB_SLAM3::GeometricCamera *mpCamera, cv::Mat mK);
void ComputeImageBounds(const cv::Mat &imLeft, cv::Mat &mDistCoef, ORB_SLAM3::GeometricCamera *mpCamera, cv::Mat &mK,
                        float &mnMinX, float &mnMaxX, float &mnMinY, float &mnMaxY);
void AssignFeaturesToGrid(int N, std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                          int Nleft, std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                          std::vector<cv::KeyPoint> mvKeysUn,
                          std::vector<cv::KeyPoint> mvKeys,
                          std::vector<cv::KeyPoint> mvKeysRight,
                          float mnMinX, float mnMinY,
                          float mfGridElementWidthInv, float mfGridElementHeightInv
);
bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY, float mnMinX, float mnMinY,
               float mfGridElementWidthInv, float mfGridElementHeightInv);

#endif //EXTRACTORB_INC_FRAME_CHANGED_H_
