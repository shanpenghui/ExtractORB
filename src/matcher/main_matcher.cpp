//
// Created by sph on 2020/11/5.
//
#include "ORBextractor.h"
#include "Frame.h"
#include "KannalaBrandt8.h"
#include <iostream>
#include <opencv2/features2d.hpp>
#include "Initializer.h"
#include "ORBmatcher.h"

using namespace std;

bool match(const cv::Mat &descriptors1, const cv::Mat &descriptors2,std::vector<cv::DMatch> &result) {

    //Simple OpenCV BF matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING2);
    matcher.match(descriptors1,descriptors2,result);
    if (result.size() < 10)
    {
        std::cout<<"Matches too less !"<<std::endl;
        return false;
    } else{
        return true;
    }
}

void match_bf(const cv::Mat& img1,const cv::Mat& img2)
{
    cv::Mat des1,des2;

    //Feature Detect
    vector<cv::KeyPoint> keypoints1,keypoints2;
    ORB_SLAM3::ORBextractor detector = ORB_SLAM3::ORBextractor(2000,1.2,8,20,7);

    vector<int> vLapping = {0, 1000};
    detector(img1,cv::Mat(),keypoints1,des1,vLapping);
    detector(img2,cv::Mat(),keypoints2,des2,vLapping);

    //BF match
    vector<cv::DMatch> simple_matches;
    vector<cv::DMatch> cross_matches;
    vector<cv::DMatch> lowe_matches;

//    ORBmatcherBF matcher;

    match(des1,des2,simple_matches);
//    matcher.match_cross_verify(des1,des2,cross_matches);
//    matcher.match_lowe_ratio(des1,des2,0.8,lowe_matches);

    string str;
    //Draw Match Reuslt
    cv::Mat simple,cross,lowe;
    cv::drawMatches(img1,keypoints1,img2,keypoints2,simple_matches,simple);
//    str="ORB_simple_bf: "+to_string(simple_matches.size());
//    cv::putText(simple,str,cv::Point(150,150),cv::FONT_HERSHEY_DUPLEX,3.0,cv::Scalar(255,255,0));

//    cv::drawMatches(img1,keypoints1,img2,keypoints2,cross_matches,cross);
//    str="ORB_cross: "+to_string(cross_matches.size());
//    cv::putText(cross,str,cv::Point(150,150),cv::FONT_HERSHEY_DUPLEX,3.0,cv::Scalar(255,255,0));

//    cv::drawMatches(img1,keypoints1,img2,keypoints2,lowe_matches,lowe);
//    str="ORB_lowe: "+to_string(lowe_matches.size());
//    cv::putText(lowe,str,cv::Point(150,150),cv::FONT_HERSHEY_DUPLEX,3.0,cv::Scalar(255,255,0));

    cv::namedWindow("ORB_simple_bf",0);
    cv::imshow("ORB_simple_bf",simple);
//    cv::namedWindow("ORB_cross",0);
//    cv::imshow("ORB_cross",cross);
//    cv::namedWindow("ORB_lowe",0);
//    cv::imshow("ORB_lowe",lowe);

    cv::waitKey(0);
}

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

    //---------------------- 1st Frame
    // 读取图像
    string image_1_file_path = "../pic/robot/866_im.jpg";
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

    // Frame construct
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

    std::vector<cv::Point2f> mvbPrevMatched;
    mvbPrevMatched.resize(mCurrentFrame_1.mvKeysUn.size());
    for(size_t i=0; i<mCurrentFrame_1.mvKeysUn.size(); i++)
        mvbPrevMatched[i]=mCurrentFrame_1.mvKeysUn[i].pt;

    ORB_SLAM3::Initializer* mpInitializer;
    mpInitializer =  new ORB_SLAM3::Initializer(mCurrentFrame_1,1.0,200);
    std::vector<int> mvIniMatches;
    fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
    cout << "mvIniMatches.size() = " << mvIniMatches.size() << endl;

    cv::Mat descriptors_1 = mCurrentFrame_1.mDescriptors;

    //---------------------- 2nd Frame
    // 读取图像
    string image_2_file_path = "../pic/robot/867_im.jpg";
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
    LOG(INFO);
    // Start Frame Constructor
    ORB_SLAM3::Frame mCurrentFrame_2;
    LOG(INFO);
    // 1.直接复制，然后把彩色图像转换成灰度图像
    cv::Mat mImGray_2 = image_2;

    // 2.从图像中回复时间戳
    double tframe_2 = 123456789;
    double timestamp_2 = tframe_2;
    LOG(INFO);
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
    LOG(INFO);

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

    cv::Mat descriptors_2 = mCurrentFrame_2.mDescriptors;

    //---------------------------------------------------------------
    // 特征匹配
    ORB_SLAM3::ORBmatcher matcher(0.9,true);
    int nmatches = matcher.SearchForInitialization(mCurrentFrame_1,mCurrentFrame_2,mvbPrevMatched,mvIniMatches,100);
    cout << "2帧匹配对数是 " << nmatches << endl;

    //---------------------------------------------------------------
    vector<cv::DMatch> matches;
    cv::BFMatcher BFmatcher;
    BFmatcher.match(descriptors_1, descriptors_2, matches);

    cout << "mCurrentFrame_1.mvKeys.size() = " << mCurrentFrame_1.mvKeys.size()
            << " mCurrentFrame_2.mvKeys.size() = " << mCurrentFrame_2.mvKeys.size()
            << " matches.size() = " << matches.size()
            << endl;
    cv::Mat out_put_match_image;
    cv::drawMatches(image_1, mCurrentFrame_1.mvKeys,
                    image_2, mCurrentFrame_2.mvKeys,
                    matches, out_put_match_image);
    cv::namedWindow("ORB drawMatches",0);
    cv::imshow("ORB drawMatches", out_put_match_image);
    cv::waitKey(0);

    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    std::vector<cv::Point3f> mvIniP3D;
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
    LOG(INFO);
    if(mpCamera->ReconstructWithTwoViews(mCurrentFrame_1.mvKeysUn,mCurrentFrame_2.mvKeysUn,
                                         mvIniMatches,Rcw,tcw,mvIniP3D,vbTriangulated)){
        cout << "ReconstructWithTwoViews success" << endl;
    }
    else {
        cout << "ReconstructWithTwoViews failed" << endl;
    }
    LOG(INFO);
//    return 0;

    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vMatches = mvIniMatches;

    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<cv::KeyPoint> mvIniKeys;
    mvIniKeys = mCurrentFrame_1.mvKeys;
    vIniKeys = mvIniKeys;

    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<cv::KeyPoint> mvCurrentKeys;
    mvCurrentKeys = mCurrentFrame_2.mvKeys;
    vCurrentKeys = mvCurrentKeys;
    for(unsigned int i=0; i<vMatches.size(); i++)
    {
        if(vMatches[i]>=0)
        {
            cv::line(image_2, vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                     cv::Scalar(0,255,0));
        }
    }
    cv::waitKey(0);

    return 0;
}
