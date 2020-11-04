//
// Created by sph on 2020/11/3.
//

#include "KannalaBrandt8.h"
#include "ORBExtractor.h"
#include "Pinhole.h"
#include "Frame_changed.h"

int main(int argc, char **argv) {

    // Google log
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "../log/log_");
    google::SetStderrLogging(google::GLOG_WARNING);

    // 读取图像
    string image_file_path = "../pic/TUM/dataset-room4_512_16/mav0/cam0/data/1520531124150444163.png";
    cv::Mat image;
    image = cv::imread(image_file_path, cv::IMREAD_GRAYSCALE);
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

    /////////////////////////////////////////////////////////////
    //- Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正  -//
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    // Monocular
    vector<int> vLapping = {0, 1000};
    ORBextractor *mpORBextractorLeft = &my_orb_extractor;
    int monoLeft = (*mpORBextractorLeft)(image, cv::Mat(), mvKeys, mDescriptors, vLapping);
//    LOG(INFO) << "monoLeft = " << monoLeft;
//    LOG(INFO) << "mvKeys.size() = " << mvKeys.size();

    //求出特征点的个数
    int N = mvKeys.size();
//    cout << "求出特征点的个数 = " << N << endl;

    //如果没有能够成功提取出特征点，那么就直接返回了
    if (mvKeys.empty()) {
        cout << "没有能够成功提取出特征点" << endl;
        return 0;
    }

//    因为tum_512数据集里面是鱼眼摄像头，所以相机配置是KannalaBrandt8

//    视觉SLAM十四讲 第五章
//    为更好地理解径向畸变和切向畸变，我们用更严格的数学形式对两者进行描述。我们知道，平面上的任意一点p 可以用笛卡儿坐标表示为[x,y ]T
//    ，也可以把它写成极坐标的形式[r,θ ]T ，其中r 表示点p 与坐标系原点之间的距离，θ 表示与水平轴的夹角。
//    径向畸变可看成坐标点沿着长度方向发生了变化δr ，也就是其距离原点的长度发生了变化。
//    切向畸变可以看成坐标点沿着切线方向发生了变化，也就是水平夹角发生了变化δθ 。

//    针孔相机模型畸变系数  k1:1阶径向畸变系数 k2:2阶径向畸变系数 k3:3阶径向畸变系数 p1:1阶切向畸变系数 p2:2阶切向畸变系数

//    我们知道，普通相机成像遵循的是针孔相机模型，在成像过程中实际场景中的直线仍被投影为图像平面上的直线。
//    但是鱼眼相机如果按照针孔相机模型成像的话，投影图像会变得非常大，当相机视场角达到180°时，图像甚至会变为无穷大。
//    所以，鱼眼相机的投影模型为了将尽可能大的场景投影到有限的图像平面内，允许了相机畸变的存在。
//    并且由于鱼眼相机的径向畸变非常严重，所以
//    鱼眼相机主要的是考虑径向畸变，而忽略其余类型的畸变!!!!

    cv::Mat mDistCoef;
    mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    mDistCoef.at<float>(0) = 0;//k1 1阶径向畸变系数
    mDistCoef.at<float>(1) = 0;//k2 2阶径向畸变系数
    mDistCoef.at<float>(2) = 0;//k3 3阶径向畸变系数
    mDistCoef.at<float>(3) = 0;//k4 4阶径向畸变系数

    // 本次验证主要是基于ORB-SLAM3的Monocular,数据集是dataset-room4_512_16,验证了跑该数据集不需要进行畸变校正
    // TODO:没明白为什么不需要畸变校正？？？
    // 具体代码参见 https://github.com/shanpenghui/ORB_SLAM3_Fixed.git
    // 分支branch是work_check_undistort

    // ORB-SLAM3中的Tracking::ParseCamParamFile(cv::FileStorage &fSettings)函数
    // if的条件为sCameraName == "KannalaBrandt8"的时候
    ORB_SLAM3::GeometricCamera *pCamera;
    float fx, fy, cx, cy;
    fx = 190.978477;
    fy = 190.973307;
    cx = 254.931706;
    cy = 256.897442;
    float k1, k2, k3, k4;
    k1 = 0.003482389402;
    k2 = 0.000715034845;
    k3 = -0.002053236141;
    k4 = 0.000202936736;
    vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
    pCamera = new ORB_SLAM3::KannalaBrandt8(vCamCalib);
    cv::Mat mK;
    mK = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = fx;
    mK.at<float>(1, 1) = fy;
    mK.at<float>(0, 2) = cx;
    mK.at<float>(1, 2) = cy;

    // 其实这里没进行畸变校正，但是为了对比ORB-SLAM3源码，还是放在这里比较好对比一些
    std::vector<cv::KeyPoint> mvKeysUn;
    cout << "mvKeys.size() = " << mvKeys.size() << endl;
    UndistortKeyPoints(mDistCoef, mvKeys, mvKeysUn, N, pCamera, mK);

    cout << "mvKeysUn.size() = " << mvKeysUn.size() << endl;
//    for (int kI = 0; kI < mvKeysUn.size(); ++kI) {
//    }

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    // Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    ComputeImageBounds(image, mDistCoef, pCamera, mK, mnMinX, mnMaxX, mnMinY, mnMaxY);
    cout << "mnMinX = " << mnMinX
         << " mnMaxX = " << mnMaxX
         << " mnMinY = " << mnMinY
         << " mnMaxY = " << mnMaxY
         << endl;

    float mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);// 0.125
//    cout << "mfGridElementWidthInv = " << mfGridElementWidthInv << endl;
    float mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);//0.09375
//    cout << "mfGridElementHeightInv = " << mfGridElementHeightInv << endl;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    std::vector<cv::KeyPoint> mvKeysRight;
    int Nleft = -1;

    // TODO: Check if it is correct!
//    AssignFeaturesToGrid(N, mGrid, Nleft, mGridRight, mvKeysUn, mvKeys,
//                         mvKeysRight, mnMinX, mnMinY, mfGridElementWidthInv, mfGridElementHeightInv);

    return 0;
}
