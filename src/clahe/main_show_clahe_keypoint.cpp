//
// Created by sph on 2020/11/5.
//

#include "ORBExtractor.h"
#include "time.h"

int main(int argc, char **argv) {

    // Google log
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "../log/log_");
    google::SetStderrLogging(google::GLOG_WARNING);

    // 读取图像
    string image_file_path = "../pic/robot/2196_im.jpg";
    cv::Mat image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    cv::Mat im_clahe;
    clock_t start = clock();
    clahe->apply(image, im_clahe);
    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "duration = " << duration << endl;

    if (im_clahe.empty()) {
        cout << "The " << image_file_path << " was not found, please check if it existed." << endl;
        return 0;
    } else
        cout << "The " << image_file_path << " Image load successed!" << endl;

    // ORB_SLAM3 yaml 配置
    int nFeatures = 1500;       // 特征点上限
    float fScaleFactor = 1.2;   // 图像金字塔缩放系数
    int nLevels = 8;            // 图像金字塔层数
    int fIniThFAST = 20;        // 默认FAST角点检测阈值
    int fMinThFAST = 7;         // 最小FAST角点检测阈值

    // 构造 ORBextractor
    ORBextractor my_orb_extractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);




    /////////////////////////////////////////////////////////////
    my_orb_extractor.ComputePyramid(image);
    vector<vector<KeyPoint>> allKeypoints_raw;
    my_orb_extractor.ComputeKeyPointsOctTree(allKeypoints_raw);

    //统计所有层的特征点并进行尺度恢复
    int image_total_keypoints_raw = 0;
    for (int level = 0; level < nLevels; ++level) {
        image_total_keypoints_raw += (int) allKeypoints_raw[level].size();
    }
    cout << "ORB_SLAM3 raw has total " << image_total_keypoints_raw << " keypoints" << endl;
    vector<cv::KeyPoint> out_put_all_keypoints_raw(image_total_keypoints_raw);
    for (int level = 0; level < nLevels; ++level) {
        if (level == 0) {
            for (int i = 0; i < allKeypoints_raw[level].size(); ++i) {
                out_put_all_keypoints_raw.push_back(allKeypoints_raw[level][i]);
            }
        }
        float scale = my_orb_extractor.mvScaleFactor[level];
        for (vector<cv::KeyPoint>::iterator key = allKeypoints_raw[level].begin();
             key != allKeypoints_raw[level].end(); key++) {
            key->pt *= scale; //尺度恢复
        }
        out_put_all_keypoints_raw.insert(out_put_all_keypoints_raw.end(), allKeypoints_raw[level].begin(),
                                     allKeypoints_raw[level].end());
    }

    cout << "最终特征点分布：" << endl;
    Mat out_put_image_raw;
    drawKeypoints(image, out_put_all_keypoints_raw, out_put_image_raw);
    imshow("ORB_SLAM3 raw", out_put_image_raw);





    /////////////////////////////////////////////////////////////
    my_orb_extractor.ComputePyramid(im_clahe);
    vector<vector<KeyPoint>> allKeypoints;
    my_orb_extractor.ComputeKeyPointsOctTree(allKeypoints);

    //统计所有层的特征点并进行尺度恢复
    int image_total_keypoints = 0;
    for (int level = 0; level < nLevels; ++level) {
        image_total_keypoints += (int) allKeypoints[level].size();
    }
    cout << "ORB_SLAM3 clahe has total " << image_total_keypoints << " keypoints" << endl;
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
    drawKeypoints(im_clahe, out_put_all_keypoints, out_put_image);
    imshow("ORB_SLAM3 clahe", out_put_image);





    /////////////////////////////////////////////////////////////
    cv::Mat img2;
    vector<cv::KeyPoint> fast_keypoints;
    Ptr<ORB> orb = ORB::create(nFeatures);
    orb->detect(im_clahe, fast_keypoints);
    drawKeypoints(im_clahe, fast_keypoints, img2);
    cout << "OpenCV orb->detect has total " << fast_keypoints.size() << " keypoints" << endl;
    cv::imshow("OpenCV orb->detect keypoints", img2);
    cv::waitKey(0);


    return 0;
}

