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

    // ORB_SLAM3 yaml 配置
    int nFeatures = 1500;       // 特征点上限
    float fScaleFactor = 1.2;   // 图像金字塔缩放系数
    int nLevels = 8;            // 图像金字塔层数
    int fIniThFAST = 20;        // 默认FAST角点检测阈值
    int fMinThFAST = 7;         // 最小FAST角点检测阈值

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    // 1st Frame
    // 读取图像
    string image_1_file_path = "../pic/robot/2195_im.jpg";
    cv::Mat image_1 = cv::imread(image_1_file_path, cv::IMREAD_GRAYSCALE);
    if (image_1.empty()) {
        cout << "The " << image_1_file_path << " was not found, please check if it existed." << endl;
        return 0;
    } else
        cout << "The " << image_1_file_path << " image_1 load successed!" << endl;

    ORB_SLAM3::ORBextractor *mpIniORBextractor = new ORB_SLAM3::ORBextractor(5 * nFeatures, fScaleFactor, nLevels,
                                                                             fIniThFAST, fMinThFAST);

    // Extract Keypoints
    mpIniORBextractor->ComputePyramid(image_1);
    vector<vector<cv::KeyPoint>> allKeypoints_1;
    mpIniORBextractor->ComputeKeyPointsOctTree(allKeypoints_1);

    // 统计所有层的特征点并进行尺度恢复
    int image_1_total_keypoints = 0;
    for (int level = 0; level < nLevels; ++level) {
        image_1_total_keypoints += (int) allKeypoints_1[level].size();
    }
    cout << "Image_1 has total " << image_1_total_keypoints << " keypoints" << endl;

    // Start Frame Constructor
    ORB_SLAM3::Frame mCurrentFrame_1;

    // 1.直接复制，然后把彩色图像转换成灰度图像
    cv::Mat mImGray_1 = image_1;

    // 2.从图像中回复时间戳
    double tframe_1 = 123456789;
    double timestamp_1 = tframe_1;

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

    // Frame 1
    mCurrentFrame_1 = ORB_SLAM3::Frame(mImGray_1, timestamp_1, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef,
                                     mbf, mThDepth);

    if(mCurrentFrame_1.mvKeys.size()>100) {
        LOG(INFO) << "mCurrentFrame_1.mvKeys.size() = " << mCurrentFrame_1.mvKeys.size();
    }
    else{
        cout << "mCurrentFrame_1.mvKeys.size() <=100 Exit 0 " << endl;
        return -1;
    }

    // Show Frame 1
    vector<cv::KeyPoint> out_put_all_keypoints_1(image_1_total_keypoints);
    for (int level = 0; level < nLevels; ++level) {
        if (level == 0) {
            for (int i = 0; i < allKeypoints_1[level].size(); ++i) {
                out_put_all_keypoints_1.push_back(allKeypoints_1[level][i]);
            }
        }
        float scale = mpIniORBextractor->mvScaleFactor[level];
        for (vector<cv::KeyPoint>::iterator key = allKeypoints_1[level].begin();
             key != allKeypoints_1[level].end(); key++) {
            key->pt *= scale; //尺度恢复
        }
        out_put_all_keypoints_1.insert(out_put_all_keypoints_1.end(), allKeypoints_1[level].begin(),
                                     allKeypoints_1[level].end());
    }

    cv::Mat out_put_image_1;
    drawKeypoints(image_1, out_put_all_keypoints_1, out_put_image_1);
    imshow("Image 1", out_put_image_1);

    // 2st Frame
    // 读取图像
    string image_2_file_path = "../pic/robot/2196_im.jpg";
    cv::Mat image_2 = cv::imread(image_2_file_path, cv::IMREAD_GRAYSCALE);
    if (image_2.empty()) {
        cout << "The " << image_2_file_path << " was not found, please check if it existed." << endl;
        return 0;
    } else
        cout << "The " << image_2_file_path << " image_2 load successed!" << endl;

    // Extract Keypoints
    mpIniORBextractor->ComputePyramid(image_2);
    vector<vector<cv::KeyPoint>> allKeypoints_2;
    mpIniORBextractor->ComputeKeyPointsOctTree(allKeypoints_2);

    // 统计所有层的特征点并进行尺度恢复
    int image_2_total_keypoints = 0;
    for (int level = 0; level < nLevels; ++level) {
        image_2_total_keypoints += (int) allKeypoints_2[level].size();
    }
    cout << "Image_2 has total " << image_2_total_keypoints << " keypoints" << endl;

    // Start Frame Constructor
    ORB_SLAM3::Frame mCurrentFrame_2;

    // 1.直接复制，然后把彩色图像转换成灰度图像
    cv::Mat mImGray_2 = image_2;

    // 2.从图像中回复时间戳
    double tframe_2 = 123456789;
    double timestamp_2 = tframe_2;

    // Frame 2
    mCurrentFrame_2 = ORB_SLAM3::Frame(mImGray_2, timestamp_2, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef,
                                       mbf, mThDepth);

    if(mCurrentFrame_2.mvKeys.size()>100) {
        LOG(INFO) << "mCurrentFrame_2.mvKeys.size() = " << mCurrentFrame_2.mvKeys.size();
    }
    else{
        cout << "mCurrentFrame_2.mvKeys.size() <=100 Exit 0 " << endl;
        return -1;
    }

    // Show Frame 2
    vector<cv::KeyPoint> out_put_all_keypoints_2(image_2_total_keypoints);
    for (int level = 0; level < nLevels; ++level) {
        if (level == 0) {
            for (int i = 0; i < allKeypoints_2[level].size(); ++i) {
                out_put_all_keypoints_2.push_back(allKeypoints_2[level][i]);
            }
        }
        float scale = mpIniORBextractor->mvScaleFactor[level];
        for (vector<cv::KeyPoint>::iterator key = allKeypoints_2[level].begin();
             key != allKeypoints_2[level].end(); key++) {
            key->pt *= scale; //尺度恢复
        }
        out_put_all_keypoints_2.insert(out_put_all_keypoints_2.end(), allKeypoints_2[level].begin(),
                                       allKeypoints_2[level].end());
    }

    cv::Mat out_put_image_2;
    drawKeypoints(image_2, out_put_all_keypoints_2, out_put_image_2);
    cv::imshow("Image 2", out_put_image_2);

    cv::waitKey(0);

    return 0;
}
