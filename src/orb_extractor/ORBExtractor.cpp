//
// Created by sph on 2020/8/3.
//

#include "ORBExtractor.h"

std::vector<float> mvLevelSigma2;
std::vector<float> mvInvLevelSigma2;

//特征点提取器的构造函数
ORBextractor::ORBextractor(int _nfeatures,        //指定要提取的特征点数目
                           float _scaleFactor,    //指定图像金字塔的缩放系数
                           int _nlevels,        //指定图像金字塔的层数
                           int _iniThFAST,        //指定初始的FAST特征点提取参数，可以提取出最明显的角点
                           int _minThFAST) :        //如果因为图像纹理不丰富提取出的特征点不多，为了达到想要的特征点数目，
//就使用这个参数提取出不是那么明显的角点
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        iniThFAST(_iniThFAST), minThFAST(_minThFAST)//设置这些参数
{
//    cout << " 特征点上限 nFeatures = " << nfeatures
//        << " 图像金字塔缩放系数 fScaleFactor = " << scaleFactor
//        << " 图像金字塔层数 nlevels = " << nlevels
//        << " 默认FAST角点检测阈值 iniThFAST = " << iniThFAST
//        << " 最小FAST角点检测阈值 minThFAST = " << minThFAST
//        << endl;

    //存储每层图像缩放系数的vector调整为符合图层数目的大小
    mvScaleFactor.resize(nlevels);
    //存储这个sigma^2，其实就是每层图像相对初始图像缩放因子的平方
    mvLevelSigma2.resize(nlevels);
    //对于初始图像，这两个参数都是1
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    //然后逐层计算图像金字塔中图像相当于初始图像的缩放系数
    for (int i = 1; i < nlevels; i++) {
        //呐，其实就是这样累乘计算得出来的
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        //原来这里的sigma^2就是每层图像相对于初始图像缩放因子的平方
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    //接下来的两个向量保存上面的参数的倒数，操作都是一样的就不再赘述了
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++) {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    //调整图像金字塔vector以使得其符合咱们设定的图像层数
    mvImagePyramid.resize(nlevels);

    //每层需要提取出来的特征点个数，这个向量也要根据图像金字塔设定的层数进行调整
    mnFeaturesPerLevel.resize(nlevels);

    //图片降采样缩放系数的倒数
    float factor = 1.0f / scaleFactor;
    //每个单位缩放系数所希望的特征点个数
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float) pow((double) factor, (double) nlevels));
//    cout << "nDesiredFeaturesPerScale = " << nDesiredFeaturesPerScale << endl;
    //用于在特征点个数分配的，特征点的累计计数清空
    int sumFeatures = 0;
    //开始逐层计算要分配的特征点个数，顶层图像除外（看循环后面）
    for (int level = 0; level < nlevels - 1; level++) {
        //分配 cvRound : 返回个参数最接近的整数值
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        //累计
        sumFeatures += mnFeaturesPerLevel[level];
        //乘系数
        nDesiredFeaturesPerScale *= factor;
    }
    //由于前面的特征点个数取整操作，可能会导致剩余一些特征点个数没有被分配，所以这里就将这个余出来的特征点分配到最高的图层中
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    //成员变量pattern的长度，也就是点的个数，这里的512表示512个点（上面的数组中是存储的坐标所以是256*2*2）
    const int npoints = 512;
    //获取用于计算BRIEF描述子的随机采样点点集头指针
    //注意到pattern0数据类型为Points*,bit_pattern_31_是int[]型，所以这里需要进行强制类型转换
    const Point *pattern0 = (const Point *) bit_pattern_31_;
    //使用std::back_inserter的目的是可以快覆盖掉这个容器pattern之前的数据
    //其实这里的操作就是，将在全局变量区域的、int格式的随机采样点以cv::point格式复制到当前类对象中的成员变量中
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    //下面的内容是和特征点的旋转计算有关的
    // pre-compute the end of a row in a circular patch
    //预先计算圆形patch中行的结束位置
    //+1中的1表示那个圆的中间行
    umax.resize(HALF_PATCH_SIZE + 1);

    //cvFloor返回不大于参数的最大整数值，cvCeil返回不小于参数的最小整数值，cvRound则是四舍五入
    int v,        //循环辅助变量
    v0,        //辅助变量
    vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);    //计算圆的最大行号，+1应该是把中间行也给考虑进去了
    //NOTICE 注意这里的最大行号指的是计算的时候的最大行号，此行的和圆的角点在45°圆心角的一边上，之所以这样选择
    //是因为圆周上的对称特性

    //这里的二分之根2就是对应那个45°圆心角

    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    //半径的平方
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;

    //利用圆的方程计算每行像素的u坐标边界（max）
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));        //结果都是大于0的结果，表示x坐标在这一行的边界

    // Make sure we are symmetric
    //这里其实是使用了对称的方式计算上四分之一的圆周上的umax，目的也是为了保持严格的对称（如果按照常规的想法做，由于cvRound就会很容易出现不对称的情况，
    //同时这些随机采样的特征点集也不能够满足旋转之后的采样不变性了）
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

/**
 * 构建图像金字塔
 * @param image 输入原图像，这个输入图像所有像素都是有效的，也就是说都是可以在其上提取出FAST角点的
 */
void ORBextractor::ComputePyramid(cv::Mat image) {
    //开始遍历所有的图层
    for (int level = 0; level < nlevels; ++level) {
        //获取本层图像的缩放系数
        float scale = mvInvScaleFactor[level];
//        cout << "本层图像的缩放系数 = " << scale << endl;
        //计算本层图像的像素尺寸大小
        Size sz(cvRound((float) image.cols * scale), cvRound((float) image.rows * scale));
        //全尺寸图像。包括无效图像区域的大小。将图像进行“补边”，EDGE_THRESHOLD区域外的图像不进行FAST角点检测
        Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        //?声明两个临时变量，temp貌似并未使用，masktemp并未使用
        Mat temp(wholeSize, image.type()), masktemp;
        //把图像金字塔该图层的图像copy给temp（这里为浅拷贝，内存相同）
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        //计算第0层以上resize后的图像
        if (level != 0) {
            //将上一层金字塔图像根据设定sz缩放到当前层级
            resize(mvImagePyramid[level - 1],    //输入图像
                   mvImagePyramid[level],    //输出图像
                   sz,                        //输出图像的尺寸
                   0,                        //水平方向上的缩放系数，留0表示自动计算
                   0,                        //垂直方向上的缩放系数，留0表示自动计算
                   cv::INTER_LINEAR);        //图像缩放的差值算法类型，这里的是线性插值算法

            //把源图像拷贝到目的图像的中央，四面填充指定的像素。图片如果已经拷贝到中间，只填充边界
            //TODO 貌似这样做是因为在计算描述子前，进行高斯滤波的时候，图像边界会导致一些问题，说不明白
            //EDGE_THRESHOLD指的这个边界的宽度，由于这个边界之外的像素不是原图像素而是算法生成出来的，所以不能够在EDGE_THRESHOLD之外提取特征点
            copyMakeBorder(mvImagePyramid[level],                    //源图像
                           temp,                                    //目标图像（此时其实就已经有大了一圈的尺寸了）
                           EDGE_THRESHOLD, EDGE_THRESHOLD,            //top & bottom 需要扩展的border大小
                           EDGE_THRESHOLD, EDGE_THRESHOLD,            //left & right 需要扩展的border大小
                           BORDER_REFLECT_101 + BORDER_ISOLATED);     //扩充方式，opencv给出的解释：

            /*Various border types, image boundaries are denoted with '|'
            * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
            * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
            * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
            * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
            * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
            */

            //BORDER_ISOLATED	表示对整个图像进行操作
            // https://docs.opencv.org/3.4.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36

        } else {
            //对于底层图像，直接就扩充边界了
            //?temp 是在循环内部新定义的，在该函数里又作为输出，并没有使用啊！
            copyMakeBorder(image,            //这里是原图像
                           temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);
        }
    }

}

/**
 * @brief 这个函数用于计算特征点的方向，这里是返回角度作为方向。
 * 计算特征点方向是为了使得提取的特征点具有旋转不变性。
 * 方法是灰度质心法：以几何中心和灰度质心的连线作为该特征点方向
 * @param[in] image     要进行操作的某层金字塔图像
 * @param[in] pt        当前特征点的坐标
 * @param[in] u_max     图像块的每一行的坐标边界 u_max
 * @return float        返回特征点的角度，范围为[0,360)角度，精度为0.3°
 */
// 参考 https://blog.csdn.net/yys2324826380/article/details/105181945/
/*
 * 灰度质心法原理：
 * 1.计算图像矩：
 *   选择某个图像块B,然后将图像块B的矩Mpq定义为:
 *           Mpq = ∑ X^pY^qI(x,y), p,q = {0,1}
 *   X,Y表示像素坐标，I(x,y)表示此像素坐标的灰度值
 * 2.图像块B的质心C可通过一下公式计算：
 *           C = (m10/m00, m01/m00)
 * 3.方向向量OC可通过将图像块B的几何中心O和它的质心C连接在一起得到：
 *           theta = arctan(m01/m10)
 * */
/* 这里有个很重要的地方是，一般我们图像块B选取的是一个矩形块，而程序中实际上选取的是一个圆形块。
 * 之所以选取圆形块，笔者认为是因为只有选取圆形块，才能保证此关键点的旋转不变性。
 * 我们可以想象一下，一个矩形块绕中心旋转任意角度，不能保证所有角度下，旋转前后两个矩形块完全重合。
 * 而圆形块绕圆心无论怎样旋转，前后圆形块一定完全重合。这就可以保证，同一关键点在图片发生旋转后，参与计算方向角的像素点与旋转前完全一样
 * 所以说ORB_SLAM的特征点带有旋转不变性
 * */
static float IC_Angle(const Mat &image, Point2f pt, const vector<int> &u_max) {
    //图像的矩，前者是按照图像块的y坐标加权，后者是按照图像块的x坐标加权
    int m_01 = 0, m_10 = 0;

    //获得这个特征点所在的图像块的中心点坐标灰度值的指针center
    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    //这条v=0中心线的计算需要特殊对待
    //由于是中心行+若干行对，所以PATCH_SIZE应该是个奇数
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        //注意这里的center下标u可以是负的！中心水平线上的像素按x坐标（也就是u坐标）加权
        m_10 += u * center[u];

    // Go line by line in the circular patch
    //这里的step1表示这个图像一行包含的字节总数。参考[https://blog.csdn.net/qianqing13579/article/details/45318279]
    int step = (int) image.step1();
    //注意这里是以v=0中心线为对称轴，然后对称地每成对的两行之间进行遍历，这样处理加快了计算速度
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
        // Proceed over the two lines
        //本来m_01应该是一列一列地计算的，但是由于对称以及坐标x,y正负的原因，可以一次计算两行
        int v_sum = 0;
        // 获取某行像素横坐标的最大范围，注意这里的图像块是圆形的！
        int d = u_max[v];
        //在坐标范围内挨个像素遍历，实际是一次遍历2个
        // 假设每次处理的两个点坐标，中心线下方为(x,y),中心线上方为(x,-y)
        // 对于某次待处理的两个点：m_10 = Σ x*I(x,y) =  x*I(x,y) + x*I(x,-y) = x*(I(x,y) + I(x,-y))
        // 对于某次待处理的两个点：m_01 = Σ y*I(x,y) =  y*I(x,y) - y*I(x,-y) = y*(I(x,y) - I(x,-y))
        for (int u = -d; u <= d; ++u) {
            //得到需要进行加运算和减运算的像素灰度值
            //val_plus：在中心线下方x=u时的的像素灰度值
            //val_minus：在中心线上方x=u时的像素灰度值
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            //在v（y轴）上，2行所有像素灰度值之差
            v_sum += (val_plus - val_minus);
            //u轴（也就是x轴）方向上用u坐标加权和（u坐标也有正负符号），相当于同时计算两行
            m_10 += u * (val_plus + val_minus);
        }
        //将这一行上的和按照y坐标加权
        m_01 += v * v_sum;
    }

    //为了加快速度还使用了fastAtan2()函数，输出为[0,360)角度，精度为0.3°
    return fastAtan2((float) m_01, (float) m_10);
}

/**
 * @brief 计算特征点的方向
 * @param[in] image                 特征点所在当前金字塔的图像
 * @param[in & out] keypoints       特征点向量
 * @param[in] umax                  每个特征点所在图像区块的每行的边界 u_max 组成的vector
 */
static void computeOrientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax) {
    // 遍历所有的特征点
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {
        // 调用IC_Angle 函数计算这个特征点的方向
        keypoint->angle = IC_Angle(image,            //特征点所在的图层的图像
                                   keypoint->pt,    //特征点在这张图像中的坐标
                                   umax);            //每个特征点所在图像区块的每行的边界 u_max 组成的vector
    }
}

//计算四叉树的特征点，函数名字后面的OctTree只是说明了在过滤和分配特征点时所使用的方式
void ORBextractor::ComputeKeyPointsOctTree(
        vector<vector<KeyPoint> > &allKeypoints)    //所有的特征点，这里第一层vector存储的是某图层里面的所有特征点，
//第二层存储的是整个图像金字塔中的所有图层里面的所有特征点
{
//    LOG(INFO) << __PRETTY_FUNCTION__ << " start";

    //重新调整图像层数
    allKeypoints.resize(nlevels);

    //图像cell的尺寸，是个正方形，可以理解为边长in像素坐标
    const float W = 30;

    // 对每一层图像做处理
    //遍历所有图像
    for (int level = 0; level < nlevels; ++level) {
        //计算这层图像的坐标边界， NOTICE 注意这里是坐标边界，EDGE_THRESHOLD指的应该是可以提取特征点的有效图像边界，后面会一直使用“有效图像边界“这个自创名词
        const int minBorderX = EDGE_THRESHOLD - 3;            //这里的3是因为在计算FAST特征点的时候，需要建立一个半径为3的圆
        const int minBorderY = minBorderX;                    //minY的计算就可以直接拷贝上面的计算结果了
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

//        cout << "minBorderX = " << minBorderX
//             << " minBorderY = " << minBorderY
//             << " maxBorderX = " << maxBorderX
//             << " maxBorderY = " << maxBorderY << endl;
        //存储需要进行平均分配的特征点
        vector<cv::KeyPoint> vToDistributeKeys;
        //一般地都是过量采集，所以这里预分配的空间大小是nfeatures*10
        vToDistributeKeys.reserve(nfeatures * 10);

        //计算进行特征点提取的图像区域尺寸
        const float width = (maxBorderX - minBorderX);
//        cout << "width = " << width << endl;
        const float height = (maxBorderY - minBorderY);
//        cout << "height = " << height << endl;

        //计算网格在当前层的图像有的行数和列数
        const int nCols = width / W;
        const int nRows = height / W;
        //计算每个图像网格所占的像素行数和列数
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

//        LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << level + 1 << " 层图像的像素宽 " << width << " 高 " << height << ", 每个格子宽 "
//                  << wCell << " 高 " << hCell
//                  << " 像素, 被切割成 " << nRows << " 行 " << nCols << " 列, ";

#ifdef SHOW_DIVIDE_IMAGE
        Mat tmp = mvImagePyramid[level];
        for (int kRow = 0; kRow < nRows; ++kRow) {
          for (int kCol = 0; kCol < nCols; ++kCol) {
            int kRow_X = minBorderX + kRow*wCell;
            int kCol_Y = minBorderY + kCol*hCell;
            tmp.col(kCol_Y) = (0, 0, 0);
            tmp.row(kRow_X) = (0, 0, 0);
          }
        }
        imshow("ComputeKeyPointsOctTree", tmp);
        waitKey(0);
#endif

        //开始遍历图像网格，还是以行开始遍历的
        for (int i = 0; i < nRows; i++) {
            //计算当前网格初始行坐标
            const float iniY = minBorderY + i * hCell;
            //计算当前网格最大的行坐标，这里的+6=+3+3，即考虑到了多出来3是为了cell边界像素进行FAST特征点提取用
            //前面的EDGE_THRESHOLD指的应该是提取后的特征点所在的边界，所以minBorderY是考虑了计算半径时候的图像边界
            //目测一个图像网格的大小是25*25啊
            float maxY = iniY + hCell + 6;

            //如果初始的行坐标就已经超过了有效的图像边界了，这里的“有效图像”是指原始的、可以提取FAST特征点的图像区域
            if (iniY >= maxBorderY - 3)
                //那么就跳过这一行
                continue;
            //如果图像的大小导致不能够正好划分出来整齐的图像网格，那么就要委屈最后一行了
            if (maxY > maxBorderY)
                maxY = maxBorderY;

            //开始列的遍历
            for (int j = 0; j < nCols; j++) {
                //计算初始的列坐标
                const float iniX = minBorderX + j * wCell;
                //计算这列网格的最大列坐标，+6的含义和前面相同
                float maxX = iniX + wCell + 6;
                //判断坐标是否在图像中
                //TODO 不太能够明白为什么要-6，前面不都是-3吗
                //!BUG  正确应该是maxBorderX-3
                if (iniX >= maxBorderX - 6)
                    continue;
                //如果最大坐标越界那么委屈一下
                if (maxX > maxBorderX)
                    maxX = maxBorderX;

                // FAST提取兴趣点, 自适应阈值
                //这个向量存储这个cell中的特征点
                vector<cv::KeyPoint> vKeysCell;
                //调用opencv的库函数来检测FAST角点
                FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),    //待检测的图像，这里就是当前遍历到的图像块
                     vKeysCell,            //存储角点位置的容器
                     iniThFAST,            //检测阈值
                     true);                //使能非极大值抑制

                //如果这个图像块中使用默认的FAST检测阈值没有能够检测到角点
                if (vKeysCell.empty()) {
                    //那么就使用更低的阈值来进行重新检测
                    FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),    //待检测的图像
                         vKeysCell,        //存储角点位置的容器
                         minThFAST,        //更低的检测阈值
                         true);            //使能非极大值抑制
//                    LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << i + 1 << " 行, 第 " << j + 1 << " 列 第 " << level + 1
//                              << " 层图像特征点数量为 " << vKeysCell.size();
                }

                //当图像cell中检测到FAST角点的时候执行下面的语句
                if (!vKeysCell.empty()) {
                    //遍历其中的所有FAST角点
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++) {
                        //NOTICE 到目前为止，这些角点的坐标都是基于图像cell的，现在我们要先将其恢复到当前的【坐标边界】下的坐标
                        //这样做是因为在下面使用八叉树法整理特征点的时候将会使用得到这个坐标
                        //在后面将会被继续转换成为在当前图层的扩充图像坐标系下的坐标
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        //然后将其加入到”等待被分配“的特征点容器中
                        vToDistributeKeys.push_back(*vit);
                    }//遍历图像cell中的所有的提取出来的FAST角点，并且恢复其在整个金字塔当前层图像下的坐标
                }//当图像cell中检测到FAST角点的时候执行下面的语句
                else {
//                    LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << i + 1 << " 行, 第 " << j + 1 << " 列 第 " << level + 1
//                              << " 层图像特征点数量为 0";
                }

            }//开始遍历图像cell的列
        }//开始遍历图像cell的行

        //声明一个对当前图层的特征点的容器的引用
        vector<KeyPoint> &keypoints = allKeypoints[level];
        //并且调整其大小为欲提取出来的特征点个数（当然这里也是扩大了的，因为不可能所有的特征点都是在这一个图层中提取出来的）
        keypoints.reserve(nfeatures);

        // 根据mnFeaturesPerLevel,即该层的兴趣点数,对特征点进行剔除
        //返回值是一个保存有特征点的vector容器，含有剔除后的保留下来的特征点
        //得到的特征点的坐标，依旧是在当前图层下来讲的
        keypoints = DistributeOctTree(vToDistributeKeys,            //当前图层提取出来的特征点，也即是等待剔除的特征点
                //NOTICE 注意此时特征点所使用的坐标都是在“半径扩充图像”下的
                                      minBorderX, maxBorderX,        //当前图层图像的边界，而这里的坐标却都是在“边缘扩充图像”下的
                                      minBorderY, maxBorderY,
                                      mnFeaturesPerLevel[level],    //希望保留下来的当前层图像的特征点个数
                                      level);                        //当前层图像所在的图层

        // 显示当前层的图像和特征点
//        DisplayImageAndKeypoints(mvImagePyramid[level], level, keypoints);

        //PATCH_SIZE是对于底层的初始图像来说的，现在要根据当前图层的尺度缩放倍数进行缩放得到缩放后的PATCH大小 和特征点的方向计算有关
        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

        // Add border to coordinates and scale information
        //获取剔除过程后保留下来的特征点数目
        const int nkps = keypoints.size();
        //然后开始遍历这些特征点，恢复其在当前图层图像坐标系下的坐标
        for (int i = 0; i < nkps; i++) {
            //对每一个保留下来的特征点，恢复到相对于当前图层“边缘扩充图像下”的坐标系的坐标
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            //记录特征点来源的图像金字塔图层
            keypoints[i].octave = level;
            //记录计算方向的patch，缩放后对应的大小， 又被称作为特征点半径
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    //然后计算这些特征点的方向信息，注意这里还是分层计算的
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level],    //对应的图层的图像
                           allKeypoints[level],    //这个图层中提取并保留下来的特征点容器
                           umax);                    //以及PATCH的横坐标边界
//    LOG(INFO) << __PRETTY_FUNCTION__ << " end";
}

// 自己添加的函数，显示各层金字塔的图像和特征点提取结果
void ORBextractor::DisplayImageAndKeypoints(const Mat &inMat, const int level, const vector<KeyPoint> &inKeyPoint) {
    Mat out_put_image;
    cv::drawKeypoints(inMat, inKeyPoint, out_put_image);
    string str = to_string(level + 1) + " level keypoints";
    imshow(str, out_put_image);
    waitKey(0);
}

/**
 * @brief 使用四叉树法对一个图像金字塔图层中的特征点进行平均和分发
 *
 * @param[in] vToDistributeKeys     等待进行分配到四叉树中的特征点
 * @param[in] minX                  当前图层的图像的边界，坐标都是在“半径扩充图像”坐标系下的坐标
 * @param[in] maxX
 * @param[in] minY
 * @param[in] maxY
 * @param[in] N                     希望提取出的特征点个数
 * @param[in] level                 指定的金字塔图层，并未使用
 * @return vector<cv::KeyPoint>     已经均匀分散好的特征点vector容器
 */
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,
                                                     const int &minX,
                                                     const int &maxX,
                                                     const int &minY,
                                                     const int &maxY,
                                                     const int &N,
                                                     const int &level) {
    // Compute how many initial nodes
    // Step 1 根据宽高比确定初始节点数目
    //计算应该生成的初始节点个数，根节点的数量nIni是根据边界的宽高比值确定的，一般是1或者2
    // ! bug: 如果宽高比小于0.5，nIni=0, 后面hx会报错
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    //一个初始的节点的x方向有多少个像素
    const float hX = static_cast<float>(maxX - minX) / nIni;

    //存储有提取器节点的列表
    list<ExtractorNode> lNodes;

    //存储初始提取器节点指针的vector
    vector<ExtractorNode *> vpIniNodes;

    //然后重新设置其大小
    vpIniNodes.resize(nIni);

//    cout << "nIni = " << nIni << " hX = " << hX << endl;
    // Step 2 生成初始提取器节点
    for (int i = 0; i < nIni; i++) {
        //生成一个提取器节点
        ExtractorNode ni;

        //设置提取器节点的图像边界
        //注意这里和提取FAST角点区域相同，都是“半径扩充图像”，特征点坐标从0 开始
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);    //UpLeft
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);  //UpRight
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);                //BottomLeft
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);             //BottomRight

        //重设vkeys大小
        ni.vKeys.reserve(vToDistributeKeys.size());

        //将刚才生成的提取节点添加到列表中
        //虽然这里的ni是局部变量，但是由于这里的push_back()是拷贝参数的内容到一个新的对象中然后再添加到列表中
        //所以当本函数退出之后这里的内存不会成为“野指针”
        lNodes.push_back(ni);
        //存储这个初始的提取器节点句柄
        vpIniNodes[i] = &lNodes.back();
//        cout << " ni.UL = " << ni.UL
//             << " ni.UR = " << ni.UR
//             << " ni.BL = " << ni.BL
//             << " ni.BR = " << ni.BR
//             << " vpIniNodes.size() = " << vpIniNodes.size()
//             << endl;
    }

    //Associate points to childs
    // Step 3 将特征点分配到子提取器节点中
//    cout << "vToDistributeKeys.size() = " << vToDistributeKeys.size() << endl;
    for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
        //获取这个特征点对象
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        //按特征点的横轴位置，分配给属于那个图像区域的提取器节点（最初的提取器节点）
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    // Step 4 遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
    // ? 这个步骤是必要的吗？感觉可以省略，通过判断nIni个数和vKeys.size() 就可以吧
    list<ExtractorNode>::iterator lit = lNodes.begin();
    while (lit != lNodes.end()) {
        //如果初始的提取器节点所分配到的特征点个数为1
        if (lit->vKeys.size() == 1) {
            //那么就标志位置位，表示此节点不可再分
            lit->bNoMore = true;
            //更新迭代器
            lit++;
        }
            ///如果一个提取器节点没有被分配到特征点，那么就从列表中直接删除它
        else if (lit->vKeys.empty())
            //注意，由于是直接删除了它，所以这里的迭代器没有必要更新；否则反而会造成跳过元素的情况
            lit = lNodes.erase(lit);
        else
            //如果上面的这些情况和当前的特征点提取器节点无关，那么就只是更新迭代器
            lit++;
    }

    //结束标志位清空
    bool bFinish = false;

    //记录迭代次数，只是记录，并未起到作用
    int iteration = 0;

    //声明一个vector用于存储节点的vSize和句柄对
    //这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
    vector<pair<int, ExtractorNode *> > vSizeAndPointerToNode;

    //调整大小，这里的意思是一个初始化节点将“分裂”成为四个，当然实际上不会有那么多，这里多分配了一些只是预防万一
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    // Step 5 根据兴趣点分布,利用4叉树方法对图像进行划分区域
    while (!bFinish) {
        //更新迭代次数计数器，只是记录，并未起到作用
        iteration++;

        //保存当前节点个数，prev在这里理解为“保留”比较好
        int prevSize = lNodes.size();

        //重新定位迭代器指向列表头部
        lit = lNodes.begin();

        //需要展开的节点计数，这个一直保持累计，不清零
        int nToExpand = 0;

        //因为是在循环中，前面的循环体中可能污染了这个变量，so清空这个vector
        //这个变量也只是统计了某一个循环中的点
        //这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
        vSizeAndPointerToNode.clear();

        // 将目前的子区域进行划分
        //开始遍历列表中所有的提取器节点，并进行分解或者保留
        while (lit != lNodes.end()) {
            //如果提取器节点只有一个特征点，
            if (lit->bNoMore) {
                // If node only contains one point do not subdivide and continue
                //那么就没有必要再进行细分了
                lit++;
                //跳过当前节点，继续下一个
                continue;
            } else {
                // If more than one point, subdivide
                //如果当前的提取器节点具有超过一个的特征点，那么就要进行继续细分
                ExtractorNode n1, n2, n3, n4;

                //再细分成四个子区域
                lit->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                //如果这里分出来的子区域中有特征点，那么就将这个子区域的节点添加到提取器节点的列表中
                //注意这里的条件是，有特征点即可
                if (n1.vKeys.size() > 0) {
                    //注意这里也是添加到列表前面的
                    lNodes.push_front(n1);

                    //再判断其中子提取器节点中的特征点数目是否大于1
                    if (n1.vKeys.size() > 1) {
                        //如果有超过一个的特征点，那么“待展开的节点计数++”
                        nToExpand++;

                        //保存这个特征点数目和节点指针的信息
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));

                        //?这个访问用的句柄貌似并没有用到？
                        // lNodes.front().lit 和前面的迭代的lit 不同，只是名字相同而已
                        // lNodes.front().lit是node结构体里的一个指针用来记录节点的位置
                        // 迭代的lit 是while循环里作者命名的遍历的指针名称
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                //后面的操作都是相同的，这里不再赘述
                if (n2.vKeys.size() > 0) {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0) {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0) {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                //当这个母节点expand之后就从列表中删除它了，能够进行分裂操作说明至少有一个子节点的区域中特征点的数量是>1的
                //? 分裂方式是后加的先分裂，先加的后分裂。
                lit = lNodes.erase(lit);

                //继续下一次循环，其实这里加不加这句话的作用都是一样的
                continue;
            }//判断当前遍历到的节点中是否有超过一个的特征点
        }//遍历列表中的所有提取器节点

        // Finish if there are more nodes than required features or all nodes contain just one point
        //停止这个过程的条件有两个，满足其中一个即可：
        //1、当前的节点数已经超过了要求的特征点数
        //2、当前所有的节点中都只包含一个特征点
        if ((int) lNodes.size() >= N                //判断是否超过了要求的特征点数
            || (int) lNodes.size() == prevSize)    //prevSize中保存的是分裂之前的节点个数，如果分裂之前和分裂之后的总节点个数一样，说明当前所有的
            //节点区域中只有一个特征点，已经不能够再细分了
        {
            //停止标志置位
            bFinish = true;
        }

            // Step 6 当再划分之后所有的Node数大于要求数目时,就慢慢划分直到使其刚刚达到或者超过要求的特征点个数
            //可以展开的子节点个数nToExpand x3，是因为一分四之后，会删除原来的主节点，所以乘以3
            /**
             * //?BUG 但是我觉得这里有BUG，虽然最终作者也给误打误撞、稀里糊涂地修复了
             * 注意到，这里的nToExpand变量在前面的执行过程中是一直处于累计状态的，如果因为特征点个数太少，跳过了下面的else-if，又进行了一次上面的遍历
             * list的操作之后，lNodes.size()增加了，但是nToExpand也增加了，尤其是在很多次操作之后，下面的表达式：
             * ((int)lNodes.size()+nToExpand*3)>N
             * 会很快就被满足，但是此时只进行一次对vSizeAndPointerToNode中点进行分裂的操作是肯定不够的；
             * 理想中，作者下面的for理论上只要执行一次就能满足，不过作者所考虑的“不理想情况”应该是分裂后出现的节点所在区域可能没有特征点，因此将for
             * 循环放在了一个while循环里面，通过再次进行for循环、再分裂一次解决这个问题。而我所考虑的“不理想情况”则是因为前面的一次对vSizeAndPointerToNode
             * 中的特征点进行for循环不够，需要将其放在另外一个循环（也就是作者所写的while循环）中不断尝试直到达到退出条件。
             * */
        else if (((int) lNodes.size() + nToExpand * 3) > N) {
            //如果再分裂一次那么数目就要超了，这里想办法尽可能使其刚刚达到或者超过要求的特征点个数时就退出
            //这里的nToExpand和vSizeAndPointerToNode不是一次循环对一次循环的关系，而是前者是累计计数，后者只保存某一个循环的
            //一直循环，直到结束标志位被置位
            while (!bFinish) {
                //获取当前的list中的节点个数
                prevSize = lNodes.size();

                //Prev这里是应该是保留的意思吧，保留那些还可以分裂的节点的信息, 这里是深拷贝
                vector<pair<int, ExtractorNode *> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                //清空
                vSizeAndPointerToNode.clear();

                // 对需要划分的节点进行排序，对pair对的第一个元素进行排序，默认是从小到大排序
                // 优先分裂特征点多的节点，使得特征点密集的区域保留更少的特征点
                //! 注意这里的排序规则非常重要！会导致每次最后产生的特征点都不一样。建议使用 stable_sort
                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

                //遍历这个存储了pair对的vector，注意是从后往前遍历
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
                    ExtractorNode n1, n2, n3, n4;
                    //对每个需要进行分裂的节点进行分裂
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    //其实这里的节点可以说是二级子节点了，执行和前面一样的操作
                    if (n1.vKeys.size() > 0) {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1) {
                            //因为这里还有对于vSizeAndPointerToNode的操作，所以前面才会备份vSizeAndPointerToNode中的数据
                            //为可能的、后续的又一次for循环做准备
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0) {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0) {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0) {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    //删除母节点，在这里其实应该是一级子节点
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    //判断是是否超过了需要的特征点数？是的话就退出，不是的话就继续这个分裂过程，直到刚刚达到或者超过要求的特征点个数
                    //作者的思想其实就是这样的，再分裂了一次之后判断下一次分裂是否会超过N，如果不是那么就放心大胆地全部进行分裂（因为少了一个判断因此
                    //其运算速度会稍微快一些），如果会那么就引导到这里进行最后一次分裂
                    if ((int) lNodes.size() >= N)
                        break;
                }//遍历vPrevSizeAndPointerToNode并对其中指定的node进行分裂，直到刚刚达到或者超过要求的特征点个数

                //这里理想中应该是一个for循环就能够达成结束条件了，但是作者想的可能是，有些子节点所在的区域会没有特征点，因此很有可能一次for循环之后
                //的数目还是不能够满足要求，所以还是需要判断结束条件并且再来一次
                //判断是否达到了停止条件
                if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize)
                    bFinish = true;
            }//一直进行不进行nToExpand累加的节点分裂过程，直到分裂后的nodes数目刚刚达到或者超过要求的特征点数目
        }//当本次分裂后达不到结束条件但是再进行一次完整的分裂之后就可以达到结束条件时
    }// 根据兴趣点分布,利用4叉树方法对图像进行划分区域

//    cout << "iteration = " << iteration << endl;
    // Retain the best point in each node
    // Step 7 保留每个区域响应值最大的一个兴趣点
    //使用这个vector来存储我们感兴趣的特征点的过滤结果
    vector<cv::KeyPoint> vResultKeys;

    //调整大小为要提取的特征点数目
    vResultKeys.reserve(nfeatures);
//    cout << "lNodes.size() = " << lNodes.size() << endl;
    //遍历这个节点列表
    for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++) {
        //得到这个节点区域中的特征点容器句柄
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;

        //得到指向第一个特征点的指针，后面作为最大响应值对应的关键点
        cv::KeyPoint *pKP = &vNodeKeys[0];

        //用第1个关键点响应值初始化最大响应值
        float maxResponse = pKP->response;

        //开始遍历这个节点区域中的特征点容器中的特征点，注意是从1开始哟，0已经用过了
        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            //更新最大响应值
            if (vNodeKeys[k].response > maxResponse) {
                //更新pKP指向具有最大响应值的keypoints
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        //将这个节点区域中的响应值最大的特征点加入最终结果容器
        vResultKeys.push_back(*pKP);
    }
//    cout << "vResultKeys.size() = " << vResultKeys.size() << endl;
    //返回最终结果容器，其中保存有分裂出来的区域中，我们最感兴趣、响应值最大的特征点
    return vResultKeys;
}

/**
 * @brief 将提取器节点分成4个子节点，同时也完成图像区域的划分、特征点归属的划分，以及相关标志位的置位
 *
 * @param[in & out] n1  提取器节点1：左上
 * @param[in & out] n2  提取器节点1：右上
 * @param[in & out] n3  提取器节点1：左下
 * @param[in & out] n4  提取器节点1：右下
 */
void ExtractorNode::DivideNode(ExtractorNode &n1,
                               ExtractorNode &n2,
                               ExtractorNode &n3,
                               ExtractorNode &n4) {
    //得到当前提取器节点所在图像区域的一半长宽，当然结果需要取整
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    //Define boundaries of childs
    //下面的操作大同小异，将一个图像区域再细分成为四个小图像区块
    //n1 存储左上区域的边界
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    //用来存储在该节点对应的图像网格中提取出来的特征点的vector
    n1.vKeys.reserve(vKeys.size());

    //n2 存储右上区域的边界
    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    //n3 存储左下区域的边界
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    //n4 存储右下区域的边界
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    //遍历当前提取器节点的vkeys中存储的特征点
    for (size_t i = 0; i < vKeys.size(); i++) {
        //获取这个特征点对象
        const cv::KeyPoint &kp = vKeys[i];
        //判断这个特征点在当前特征点提取器节点图像的哪个区域，更严格地说是属于那个子图像区块
        //然后就将这个特征点追加到那个特征点提取器节点的vkeys中
        //NOTICE BUG REVIEW 这里也是直接进行比较的，但是特征点的坐标是在“半径扩充图像”坐标系下的，而节点区域的坐标则是在“边缘扩充图像”坐标系下的
        if (kp.pt.x < n1.UR.x) {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        } else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }//遍历当前提取器节点的vkeys中存储的特征点

    //判断每个子特征点提取器节点所在的图像中特征点的数目（就是分配给子节点的特征点数目），然后做标记
    //这里判断是否数目等于1的目的是确定这个节点还能不能再向下进行分裂
    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}

const float factorPI = (float) (CV_PI / 180.f);

/**
 * @brief 计算ORB特征点的描述子。注意这个是全局的静态函数，只能是在本文件内被调用
 * @param[in] kpt       特征点对象
 * @param[in] img       提取出特征点的图像
 * @param[in] pattern   预定义好的随机采样点集
 * @param[out] desc     用作输出变量，保存计算好的描述子，长度为32*8bit
 */
static void computeOrbDescriptor(const KeyPoint &kpt,
                                 const Mat &img, const Point *pattern,
                                 uchar *desc) {
    //得到特征点的角度，用弧度制表示。kpt.angle是角度制，范围为[0,360)度
    float angle = (float) kpt.angle * factorPI;
    //然后计算这个角度的余弦值和正弦值
    float a = (float) cos(angle), b = (float) sin(angle);

    //获得图像中心指针
    const uchar *center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    //获得图像的每行的字节数
    const int step = (int) img.step;

    //原始的BRIEF描述子不具有方向信息，通过加入特征点的方向来计算描述子，称之为Steer BRIEF，具有较好旋转不变特性
    //具体地，在计算的时候需要将这里选取的随机点点集的x轴方向旋转到特征点的方向。
    //获得随机“相对点集”中某个idx所对应的点的灰度,这里旋转前坐标为(x,y), 旋转后坐标(x',y')推导:
    // x'= xcos(θ) - ysin(θ),  y'= xsin(θ) + ycos(θ)
#define GET_VALUE(idx) center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + cvRound(pattern[idx].x*a - pattern[idx].y*b)]
    // y'* step
    // x'
    //brief描述子由32*8位组成
    //其中每一位是来自于两个像素点灰度的直接比较，所以每比较出8bit结果，需要16个随机点，这也就是为什么pattern需要+=16的原因
    for (int i = 0; i < 32; ++i, pattern += 16) {

        int t0,    //参与比较的一个特征点的灰度值
        t1,        //参与比较的另一个特征点的灰度值		//TODO 检查一下这里灰度值为int型？？？
        val;    //描述子这个字节的比较结果

        t0 = GET_VALUE(0);
        t1 = GET_VALUE(1);
        val = t0 < t1;                            //描述子本字节的bit0
        t0 = GET_VALUE(2);
        t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;                    //描述子本字节的bit1
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;                    //描述子本字节的bit2
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;                    //描述子本字节的bit3
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;                    //描述子本字节的bit4
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;                    //描述子本字节的bit5
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;                    //描述子本字节的bit6
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;                    //描述子本字节的bit7

        //保存当前比较的出来的描述子的这个字节
        desc[i] = (uchar) val;
    }//通过对随机点像素灰度的比较，得出BRIEF描述子，一共是32*8=256位

    //为了避免和程序中的其他部分冲突在，在使用完成之后就取消这个宏定义
#undef GET_VALUE
}

//注意这是一个不属于任何类的全局静态函数，static修饰符限定其只能够被本文件中的函数调用
/**
 * @brief 计算某层金字塔图像上特征点的描述子
 *
 * @param[in] image                 某层金字塔图像
 * @param[in] keypoints             特征点vector容器
 * @param[out] descriptors          描述子
 * @param[in] pattern               计算描述子使用的固定随机点集
 */
static void computeDescriptors(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors,
                               const vector<Point> &pattern) {
    //清空保存描述子信息的容器
    descriptors = Mat::zeros((int) keypoints.size(), 32, CV_8UC1);

    //开始遍历特征点
    for (size_t i = 0; i < keypoints.size(); i++)
        //计算这个特征点的描述子
        computeOrbDescriptor(keypoints[i],                //要计算描述子的特征点
                             image,                    //以及其图像
                             &pattern[0],                //随机点集的首地址
                             descriptors.ptr((int) i));    //提取出来的描述子的保存位置
}

/**
 * @brief 用仿函数（重载括号运算符）方法来计算图像特征点
 *
 * @param[in] _image                    输入原始图的图像
 * @param[in] _mask                     掩膜mask
 * @param[in & out] _keypoints                存储特征点关键点的向量
 * @param[in & out] _descriptors              存储特征点描述子的矩阵
 */
int ORBextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints,
                             OutputArray _descriptors, std::vector<int> &vLappingArea) {
    // Step 1 检查图像有效性。如果图像为空，那么就直接返回
    //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
    if (_image.empty())
        return -1;

    //获取图像的大小
    Mat image = _image.getMat();
    //判断图像的格式是否正确，要求是单通道灰度值
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    // Step 2 构建图像金字塔
    ComputePyramid(image);

    // Step 3 计算图像的特征点，并且将特征点进行均匀化。均匀的特征点可以提高位姿计算精度
    // 存储所有的特征点，注意此处为二维的vector，第一维存储的是金字塔的层数，第二维存储的是那一层金字塔图像里提取的所有特征点
    vector<vector<KeyPoint> > allKeypoints;
    //使用四叉树的方式计算每层图像的特征点并进行分配
    ComputeKeyPointsOctTree(allKeypoints);

    //使用传统的方法提取并平均分配图像的特征点，作者并未使用
    //ComputeKeyPointsOld(allKeypoints);


    // Step 4 拷贝图像描述子到新的矩阵descriptors
    Mat descriptors;

    //统计整个图像金字塔中的特征点
    int nkeypoints = 0;
    //开始遍历每层图像金字塔，并且累加每层的特征点个数
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int) allKeypoints[level].size();

    //如果本图像金字塔中没有任何的特征点
    if (nkeypoints == 0)
        //通过调用cv::mat类的.realse方法，强制清空矩阵的引用计数，这样就可以强制释放矩阵的数据了
        //参考[https://blog.csdn.net/giantchen547792075/article/details/9107877]
        _descriptors.release();
    else {
        //如果图像金字塔中有特征点，那么就创建这个存储描述子的矩阵，注意这个矩阵是存储整个图像金字塔中特征点的描述子的
        _descriptors.create(nkeypoints, 32, CV_8U);
        //获取这个描述子的矩阵信息
        // ?为什么不是直接在参数_descriptors上对矩阵内容进行修改，而是重新新建了一个变量，复制矩阵后，在这个新建变量的基础上进行修改？
        descriptors = _descriptors.getMat();
    }

    //_keypoints.clear();
    //_keypoints.reserve(nkeypoints);
    _keypoints = vector<cv::KeyPoint>(nkeypoints);

    //因为遍历是一层一层进行的，但是描述子那个矩阵是存储整个图像金字塔中特征点的描述子，所以在这里设置了Offset变量来保存“寻址”时的偏移量，
    //辅助进行在总描述子mat中的定位
    int offset = 0;
    //Modified for speeding up stereo fisheye matching
    int monoIndex = 0, stereoIndex = nkeypoints - 1;
    //开始遍历每一层图像
//    cout << "nlevels = " << nlevels << endl;
    for (int level = 0; level < nlevels; ++level) {
        //获取在allKeypoints中当前层特征点容器的句柄
        vector<KeyPoint> &keypoints = allKeypoints[level];
        //本层的特征点数
        int nkeypointsLevel = (int) keypoints.size();

        //如果特征点数目为0，跳出本次循环，继续下一层金字塔
        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image
        //  Step 5 对图像进行高斯模糊
        // 深拷贝当前金字塔所在层级的图像
        Mat workingMat = mvImagePyramid[level].clone();

        // 注意：提取特征点的时候，使用的是清晰的原图像；这里计算描述子的时候，为了避免图像噪声的影响，使用了高斯模糊
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors 计算描述子
        //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        // desc存储当前图层的描述子
        Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
        // Step 6 计算高斯模糊后图像的描述子
        computeDescriptors(workingMat,    //高斯模糊之后的图层图像
                           keypoints,    //当前图层中的特征点集合
                           desc,        //存储计算之后的描述子
                           pattern);    //随机采样点集

//        cout << "keypoints.size() = " << keypoints.size() << endl;
        // 更新偏移量的值
        offset += nkeypointsLevel;


        // 获取当前图层上的缩放系数
        float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
        int i = 0;
        // 遍历本层所有的特征点
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {

            // Scale keypoint coordinates
            // Step 6 对非第0层图像中的特征点的坐标恢复到第0层图像（原图像）的坐标系下
            // ? 得到所有层特征点在第0层里的坐标放到_keypoints里面
            // 对于第0层的图像特征点，他们的坐标就不需要再进行恢复了
            if (level != 0) {
                // 特征点本身直接乘缩放倍数就可以了
                keypoint->pt *= scale;
            }

            // And add the keypoints to the output
            // 将keypoints中内容插入到_keypoints 的末尾
            // keypoint其实是对allkeypoints中每层图像中特征点的引用，这样allkeypoints中的所有特征点在这里被转存到输出的_keypoints
            // ORB-SLAM2: // _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
            // ORB-SLAM3:
//            cout << "vLappingArea[0] = " << vLappingArea[0]
//                << " vLappingArea[1] = " << vLappingArea[1]
//                << endl;
            if (keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]) {
//                cout << "stereoIndex--" << endl;
                _keypoints.at(stereoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(stereoIndex));
                stereoIndex--;
            } else {
//                cout << "monoIndex++" << endl;
                _keypoints.at(monoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(monoIndex));
                monoIndex++;
            }
            i++;
        }
    }
    //cout << "[ORBextractor]: extracted " << _keypoints.size() << " KeyPoints" << endl;
    return monoIndex;
}


