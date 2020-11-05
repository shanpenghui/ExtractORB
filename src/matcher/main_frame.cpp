//
// Created by sph on 2020/11/5.
//

#include "ORBextractor.h"
#include "Frame.h"
#include "KannalaBrandt8.h"
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

    // 1.直接复制，然后把彩色图像转换成灰度图像
    cv::Mat mImGray = image;
//    cvtColor(mImGray,mImGray,CV_RGBA2GRAY);

    // 2.从图像中回复时间戳
    double tframe = 123456789;
    double timestamp = tframe;

    // 3.加载字典模型
    // 在System里面加载
    ORB_SLAM3::ORBVocabulary* mpVocabulary = new ORB_SLAM3::ORBVocabulary();
    std::string strVocFile = "../Vocabulary/ORBvoc.txt";
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    cout << "Loading Vocabulary!" << endl;
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    ORB_SLAM3::ORBVocabulary* mpORBVocabulary = mpVocabulary;
    cout << "Vocabulary loaded!" << endl << endl;

    // 在Tracking里面加载
    // 4.加载相机模型
    float fx=190.978477,fy=190.973307,
    cx=254.931706,cy=256.897442,
    k1=0.003482389402,k2=0.000715034845,
    k3=-0.002053236141,k4=0.000202936736;
    vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
    ORB_SLAM3::GeometricCamera* mpCamera = new ORB_SLAM3::KannalaBrandt8(vCamCalib);

    // 5.加载畸变校正matrix
    cv::Mat mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    mDistCoef.at<float>(0) = 0;
    mDistCoef.at<float>(0) = 0;
    mDistCoef.at<float>(0) = 0;
    mDistCoef.at<float>(0) = 0;

    // 6.相机的基线长度 * 相机的焦距
    float mbf = 0;

    // 7.用于区分远点和近点的阈值. 近点认为可信度比较高;远点则要求在两个关键帧中得到匹配
    float mThDepth = 0;

    mCurrentFrame = ORB_SLAM3::Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef,
                          mbf, mThDepth);

    if(mCurrentFrame.mvKeys.size()>100) {
        LOG(INFO) << "mCurrentFrame.mvKeys.size() = " << mCurrentFrame.mvKeys.size();
    }

    return 0;
}
