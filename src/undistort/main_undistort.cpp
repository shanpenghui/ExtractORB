//
// Created by sph on 2020/11/3.
//

#include "ORBExtractor.h"
#include "Pinhole.h"
#include "Frame.h"

int main(int argc, char **argv) {

    // Google log
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "../log/log_");
    google::SetStderrLogging(google::GLOG_WARNING);

    // 读取图像
    string image_file_path = "../pic/TUM/dataset-room4_512_16/mav0/cam0/data/1520531124150444163.png";
    cv::Mat image = cv::imread(image_file_path, CV_LOAD_IMAGE_GRAYSCALE);
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
    //- Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图 -//
    // 构造 ORBextractor
    ORBextractor my_orb_extractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    /////////////////////////////////////////////////////////////
    //- Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正  -//
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    // Monocular
    vector<int> vLapping = {0,1000};
    ORBextractor* mpORBextractorLeft = &my_orb_extractor;
    int monoLeft = (*mpORBextractorLeft)(image,cv::Mat(),mvKeys,mDescriptors,vLapping);
//    LOG(INFO) << "monoLeft = " << monoLeft;
//    LOG(INFO) << "mvKeys.size() = " << mvKeys.size();

    //求出特征点的个数
    int N = mvKeys.size();

    //如果没有能够成功提取出特征点，那么就直接返回了
    if (mvKeys.empty()) {
        cout << "没有能够成功提取出特征点" << endl;
        return 0;
    }

    cv::Mat mDistCoef;
    mDistCoef = cv::Mat::zeros(5,1,CV_32F);
    // TUM_512.yaml in Monocular folder
    mDistCoef.at<float>(0) = 0.003482389402;//k1
    mDistCoef.at<float>(1) = 0.000715034845;//k2
    mDistCoef.at<float>(2) = 0;//p1
    mDistCoef.at<float>(3) = 0;//p2
    mDistCoef.at<float>(4) = -0.002053236141;//k3

    //
    const std::vector<cv::KeyPoint> mvKeysUn;
    ORB_SLAM3::GeometricCamera *pCamera;
    // TUM_512.yaml in Monocular folder
    float fx, fy, cx, cy;
    fx = 190.978477;
    fy = 190.973307;
    cx = 254.931706;
    cy = 256.897442;
    vector<float> vCamCalib{fx,fy,cx,cy};
    pCamera = new ORB_SLAM3::Pinhole(vCamCalib);
    cv::Mat mK(static_cast<ORB_SLAM3::Pinhole*>(pCamera)->toK());
    UndistortKeyPoints(mDistCoef, mvKeys, mvKeysUn, N, pCamera, mK);

    return 0;
}
