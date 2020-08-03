//
// Created by sph on 2020/8/3.
//

#include "../inc/ORBExtractor.h"

int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);
  google::SetLogDestination(google::GLOG_INFO, "../log/log_");
  google::SetStderrLogging(google::GLOG_INFO);

  // 1.读取图像
  cv::Mat image = cv::imread("../pic/luna.jpg", CV_LOAD_IMAGE_COLOR);
  // 1.读取图像 -合法性检查
  if (image.empty()) {
    cout << "no picture was found ...." << endl;
    return 0;
  } else
    cout << "image load successed!" << endl;

#ifdef IS_ORB_SLAM3
  assert(image.type() == CV_8UC1);
#endif

  // Load ORB parameters
  // 参考 ORB_SLAM3 启动的 yaml 文件
//# ORB Extractor: Number of features per image
//  ORBextractor.nFeatures: 1500 # Tested with 1250
  int nFeatures = 1500;// fSettings["ORBextractor.nFeatures"];
//# ORB Extractor: Scale factor between levels in the scale pyramid
//  ORBextractor.scaleFactor: 1.2
  float fScaleFactor = 1.2;// fSettings["ORBextractor.scaleFactor"];
//# ORB Extractor: Number of levels in the scale pyramid
//  ORBextractor.nLevels: 8
  int nLevels = 8;// fSettings["ORBextractor.nLevels"];
//  ORBextractor.iniThFAST: 20 # 20
//  ORBextractor.minThFAST: 7 # 7
  int fIniThFAST = 20;// fSettings["ORBextractor.iniThFAST"];
  int fMinThFAST = 7;// fSettings["ORBextractor.minThFAST"];
  ORBextractor my_orb_extractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
  my_orb_extractor.ComputePyramid(image);
  vector < vector<KeyPoint> > allKeypoints;
  my_orb_extractor.ComputeKeyPointsOctTree(allKeypoints);

  //统计所有层的特征点并进行尺度恢复
  int image_total_keypoints = 0;
  for (int level = 0; level < nLevels; ++level) {
    image_total_keypoints += (int) allKeypoints[level].size();
  }
  cout << "ORB_SLAM3 has total " << image_total_keypoints << " keypoints" << endl;
  vector<cv::KeyPoint> out_put_all_keypoints(image_total_keypoints);
  for (int level = 0; level < nLevels; ++level) {
    if (level == 0) {
      for (int i = 0; i < allKeypoints[level].size(); ++i) {
        out_put_all_keypoints.push_back(allKeypoints[level][i]);
      }
    }
    float scale = my_orb_extractor.mvScaleFactor[level];
    for (vector<cv::KeyPoint>::iterator key = allKeypoints[level].begin();
         key != allKeypoints[level].end(); key++) {
      key->pt *= scale; //尺度恢复
    }
    out_put_all_keypoints.insert(out_put_all_keypoints.end(), allKeypoints[level].begin(),
                                 allKeypoints[level].end());
  }
  
  cout << "最终特征点分布：" << endl;
  Mat out_put_image;
  drawKeypoints(image, out_put_all_keypoints, out_put_image);
  imshow("ORB_SLAM3 extract keypoints", out_put_image);

  cv::Mat img2;
  vector<cv::KeyPoint> fast_keypoints;
  Ptr<ORB> orb = ORB::create(nFeatures);
  orb->detect(image, fast_keypoints);
  drawKeypoints(image, fast_keypoints, img2);
  cout << "OpenCV orb->detect has total " << fast_keypoints.size() << " keypoints" << endl;
  cv::imshow("OpenCV orb->detect keypoints", img2);
  cv::waitKey(0);

  return 0;
}