//
// Created by sph on 2020/8/3.
//

#include "ORBExtractor.h"

int main(int argc, char **argv) {

    // Google log
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "../log/log_");
    google::SetStderrLogging(google::GLOG_WARNING);

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
    //- Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图 -//
    // 构造 ORBextractor
    ORBextractor my_orb_extractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    my_orb_extractor.ComputePyramid(image);
    vector<vector<KeyPoint>> allKeypoints;
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



    /////////////////////////////////////////////////////////////
    //- Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正  -//
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    // Monocular
    vector<int> vLapping = {0, 1000};
    ORBextractor *mpORBextractorLeft = &my_orb_extractor;
    int monoLeft = (*mpORBextractorLeft)(image, cv::Mat(), mvKeys, mDescriptors, vLapping);
    LOG(INFO) << "monoLeft = " << monoLeft;
    LOG(INFO) << "mvKeys.size() = " << mvKeys.size();

    //求出特征点的个数
//    int N = mvKeys.size();

    //如果没有能够成功提取出特征点，那么就直接返回了
//    if (mvKeys.empty())
//        return;

    return 0;
}