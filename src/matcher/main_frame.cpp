//
// Created by sph on 2020/11/5.
//

#include "ORBextractor.h"
#include "Frame.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // Google log
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "../log/log_");

    // 读取图像
    string image_file_path = "../pic/TUM/dataset-room4_512_16/mav0/cam0/data/1520531124150444163.png";
    cv::Mat image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "The " << image_file_path << " was not found, please check if it existed." << endl;
        return 0;
    } else
        cout << "The " << image_file_path << " image load successed!" << endl;

    // ORB_SLAM3 yaml 配置
    int nFeatures = 1500;       // 特征点上限
    float fScaleFactor = 1.2;   // 图像金字塔缩放系数
    int nLevels = 8;            // 图像金字塔层数
    int fIniThFAST = 20;        // 默认FAST角点检测阈值
    int fMinThFAST = 7;         // 最小FAST角点检测阈值

    /////////////////////////////////////////////////////////////
    ORB_SLAM3::ORBextractor *mpIniORBextractor = new ORB_SLAM3::ORBextractor(5 * nFeatures, fScaleFactor, nLevels,
                                                                             fIniThFAST, fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    mpIniORBextractor->ComputePyramid(image);
    vector<vector<cv::KeyPoint>> allKeypoints;
    mpIniORBextractor->ComputeKeyPointsOctTree(allKeypoints);

    //统计所有层的特征点并进行尺度恢复
    int image_total_keypoints = 0;
    for (int level = 0; level < nLevels; ++level) {
        image_total_keypoints += (int) allKeypoints[level].size();
    }
    cout << "ORB_SLAM3 has total " << image_total_keypoints << " keypoints" << endl;

    ORB_SLAM3::Frame mCurrentFrame;
//    mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth,
//                          &mLastFrame, *mpImuCalib);

    return 0;
}
