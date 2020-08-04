//
// Created by sph on 2020/8/3.
//

#ifndef DISTRIBUTE_OCT_TREE__ORBEXTRACTOR_H_
#define DISTRIBUTE_OCT_TREE__ORBEXTRACTOR_H_

#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "pattern.h"

//#define SHOW_DIVIDE_IMAGE

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

using namespace std;
using namespace cv;

class ExtractorNode
{
 public:
  ExtractorNode():bNoMore(false){}

  void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class ORBextractor {
 public:
  ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
               int _iniThFAST, int _minThFAST);
  ~ORBextractor() {}
  std::vector<cv::Mat> mvImagePyramid;

  void ComputePyramid(cv::Mat image);

  void ComputeKeyPointsOctTree(vector<vector<KeyPoint> > &allKeypoints);
  vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,
                                         const int &minX,
                                         const int &maxX,
                                         const int &minY,
                                         const int &maxY,
                                         const int &N,
                                         const int &level);
  int operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                OutputArray _descriptors, std::vector<int> &vLappingArea);

  std::vector<float> mvScaleFactor;

  void DisplayImageAndKeypoints(const Mat &inMat, const int level, const vector<KeyPoint> &inKeyPoint);

 protected:
  std::vector<int> mnFeaturesPerLevel;
  std::vector<float> mvInvScaleFactor;

  std::vector<cv::Point> pattern;

  std::vector<int> umax;
  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

};


#endif //DISTRIBUTE_OCT_TREE__ORBEXTRACTOR_H_
