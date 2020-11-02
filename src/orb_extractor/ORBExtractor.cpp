//
// Created by sph on 2020/8/3.
//

#include "../../inc/ORBExtractor.h"

ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                           int _iniThFAST, int _minThFAST) :
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST) {

  LOG(INFO) << __PRETTY_FUNCTION__ << " start";

  mvScaleFactor.resize(nlevels);// nlevels(8)层金字塔的尺度因子(mvScaleFactor)
//  mvLevelSigma2.resize(nlevels);
  mvScaleFactor[0] = 1.0f;// 金字塔第一层([0])的尺度(mvScaleFactor)是1, 即原图像
//  mvLevelSigma2[0]=1.0f;
  for (int i = 1; i < nlevels; i++) {
    mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;// 根据金字塔图像之间的尺度参数计算每层金字塔的尺度因子
//    mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
  }

  mvInvScaleFactor.resize(nlevels);// nlevels(8)层金字塔的尺度因子倒数(mvInvScaleFactor)
//  mvInvLevelSigma2.resize(nlevels);
  for (int i = 0; i < nlevels; i++) {
    mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
//    mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
  }

  mvImagePyramid.resize(nlevels);// 每层金字塔的图像
  mnFeaturesPerLevel.resize(nlevels);// 每层金字塔的特征点数量

  // 计算当前层金字塔上的特征点
  // 等比数列公式 sn = a1(1-q^level)/(1-q)
  // 因为当前是反过来求, 所以 nfeatures = nDesiredFeaturesPerScale * ( 1 - q ^ level) / (1 - q);
  float factor = 1.0f / scaleFactor;
  float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float) pow((double) factor, (double) nlevels));

  int sumFeatures = 0;
  for (int level = 0; level < nlevels - 1; level++) {
    mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
    sumFeatures += mnFeaturesPerLevel[level];
    nDesiredFeaturesPerScale *= factor;
  }
  mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

  const int npoints = 512;
  const Point *pattern0 = (const Point *) bit_pattern_31_;
  std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

  //This is for orientation
  // pre-compute the end of a row in a circular patch
  umax.resize(HALF_PATCH_SIZE + 1);

  int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
  int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
  const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
  for (v = 0; v <= vmax; ++v)
    umax[v] = cvRound(sqrt(hp2 - v * v));

  // Make sure we are symmetric
  for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
  {
    while (umax[v0] == umax[v0 + 1])
      ++v0;
    umax[v] = v0;
    ++v0;
  }
  LOG(INFO) << __PRETTY_FUNCTION__ << "   end";
}

void ORBextractor::ComputePyramid(cv::Mat image) {
  for (int level = 0; level < nlevels; ++level) {
    float scale = mvInvScaleFactor[level];
    Size sz(cvRound((float) image.cols * scale), cvRound((float) image.rows * scale));
    Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
    Mat temp(wholeSize, image.type()), masktemp;
    mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

    // Compute the resized image
    if (level != 0) {
      resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

      copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     BORDER_REFLECT_101 + BORDER_ISOLATED);
    } else {
      copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
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
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
  // 图像的矩，前者是按照图像块的y坐标加权，后者是按照图像块的x坐标加权
  int m_01 = 0, m_10 = 0;

  // 获得这个特征点所在的图像块的中心点坐标灰度值的指针center
  // TODO:为啥还要获取这个指针啊？为了取中心点附近的图像像素，用 center[u] 进行索引
  // cvRound():返回跟参数最接近的整数值,即四舍五入
  // image.at<uchar>(i,j)：取出灰度图像中i行j列的点
  const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

  // Treat the center line differently, v=0
  // 这条 v=0 中心线的计算需要特殊对待
  // TODO: v = 0 是啥东西？ v=0相当于y=0，即x轴
  // 由于是中心行+若干行对，所以PATCH_SIZE应该是个奇数
  for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
    // 注意这里的center下标u可以是负的！中心水平线上的像素按x坐标（也就是u坐标）加权
    m_10 += u * center[u];

  // Go line by line in the circuI853lar patch
  // 这里的 step1 表示这个图像一行包含的字节总数。
  // 因为 center是指针地址, 如果用center进行像素坐标索引的话，以 v=0 为基准，是以行来计算，所以需要计算一行的字节总数
  // 参考[https://blog.csdn.net/qianqing13579/article/details/45318279]
  int step = (int)image.step1();
  // 注意这里是以 v = 0 中心线为对称轴，然后对称地每成对的两行之间进行遍历，这样处理加快了计算速度
  for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
  {
    // Proceed over the two lines
    // 本来m_01应该是一列一列地计算的，但是由于对称以及坐标x,y正负的原因，可以一次计算两行
    int v_sum = 0;
    // 获取某行像素横坐标的最大范围，注意这里的图像块是圆形的！
    int d = u_max[v];
    // 在坐标范围内挨个像素遍历，实际是一次遍历2个
    // 假设每次处理的两个点坐标，中心线上方为(x,y),中心线下方为(x,-y)
    // 对于某次待处理的两个点：m_10 = Σ x*I(x,y) =  x*I(x,y) + x*I(x,-y) = x*(I(x,y) + I(x,-y))
    // TODO: 为什么是 - y*I(x,-y) ？
    // 对于某次待处理的两个点：m_01 = Σ y*I(x,y) =  y*I(x,y) - y*I(x,-y) = y*(I(x,y) - I(x,-y))
    for (int u = -d; u <= d; ++u)
    {
      // 得到需要进行加运算和减运算的像素灰度值
      // val_plus：在中心线下方x=u时的的像素灰度值
      // val_minus：在中心线上方x=u时的像素灰度值
      int val_plus = center[u + v*step], val_minus = center[u - v*step];
      // 在v（y轴）上，2行所有像素灰度值之差
      v_sum += (val_plus - val_minus);
      // u轴（也就是x轴）方向上用u坐标加权和（u坐标也有正负符号），相当于同时计算两行
      m_10 += u * (val_plus + val_minus);
    }
    // 将这一行上的和按照y坐标加权
    m_01 += v * v_sum;
  }

  // 为了加快速度还使用了fastAtan2()函数，输出为[0,360)角度，精度为0.3°
  // TODO:为什么就加快了？
  return fastAtan2((float)m_01, (float)m_10);
}

/**
 * @brief 计算特征点的方向
 * @param[in] image                 特征点所在当前金字塔的图像
 * @param[in & out] keypoints       特征点向量
 * @param[in] umax                  每个特征点所在图像区块的每行的边界 u_max 组成的vector
 */
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
  // 遍历所有的特征点
  for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
           keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
  {
    // 调用IC_Angle 函数计算这个特征点的方向
    keypoint->angle = IC_Angle(image, keypoint->pt, umax);
  }
}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> > &allKeypoints) {
  LOG(INFO) << __PRETTY_FUNCTION__ << " start";

  // 初始化 nlevels 层图像金字塔的特征点容器
  allKeypoints.resize(nlevels);

  // 图像cell的尺寸，是个正方形，可以理解为边长in像素坐标
  // 设置网格参数是 30, 表示默认先把图像分割在 30 × 30的格子当中
  const float W = 30;

  for (int level = 0; level < nlevels; ++level) {
    // 计算每一层图像进行特征检测区域的边缘坐标
    const int minBorderX = EDGE_THRESHOLD - 3;// 因为FAST是检测中心点附近3个点的范围,即半径是3
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
    const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

    vector<cv::KeyPoint> vToDistributeKeys;// 存储需要进行平均分配的特征点
    vToDistributeKeys.reserve(nfeatures * 10);// 一般地都是过量采集，所以这里预分配的空间大小是nfeatures*10

    const float width = (maxBorderX - minBorderX);// 计算特征点提取的图像区域尺寸, 像素坐标
    const float height = (maxBorderY - minBorderY);

    const int nCols = width / W;// 在特征点提取的图像区域内，计算最终产生的格子行列数
    const int nRows = height / W;
    const int wCell = ceil(width / nCols);// 计算每个格子所占的像素行列数
    const int hCell = ceil(height / nRows);
    LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << level + 1 << " 层图像的像素宽 " << width << " 高 " << height << ", 每个格子宽 "
              << wCell << " 高 " << hCell
              << " 像素, 被切割成 " << nRows << " 行 " << nCols << " 列, ";

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

    for (int i = 0; i < nRows; i++) {
      // 计算行坐标初始值
      const float iniY = minBorderY + i * hCell;
      // 计算行坐标最大值，这里的+6=+3+3，即考虑到了多出来以便进行FAST特征点提取用的3像素边界
      float maxY = iniY + hCell + 6;

      // 如果初始的行坐标就已经超过了可以提取FAST特征点的图像区域
      if (iniY >= maxBorderY - 3)
        // 则跳过后面所有
        continue;
      // 如果图像的大小导致不能够正好划分出来整齐的图像网格
      if (maxY > maxBorderY) {
        // 那么多余的 maxY - maxBorderY 就不要了
        maxY = maxBorderY;
        LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << i + 1 << " 行 maxY - maxBorderY = " << maxY - maxBorderY;
      }

      for (int j = 0; j < nCols; j++) {
        // 计算列坐标初始值
        const float iniX = minBorderX + j * wCell;
        // 计算列坐标最大值
        float maxX = iniX + wCell + 6;
        // 如果列坐标初始值超过该区域,
        // TODO:bug: 这里为什么 -6， 而上面 -3
        if (iniX >= maxBorderX - 6)
          // 则跳过后面所有
          continue;
        // 如果列坐标最大值超过该区域
        if (maxX > maxBorderX) {
          // 多余的列就不要了
          maxX = maxBorderX;
          LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << i + 1 << " 行, 第 " << j + 1 << " 列 maxX - maxBorderX = "
                    << maxX - maxBorderX;
        }

        // 保存当前区域的特征点
        vector<cv::KeyPoint> vKeysCell;

        // 以默认阈值20, 对当前区域进行FAST特征点提取
        FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
             vKeysCell, iniThFAST, true);

        /*if(bRight && j <= 13){
            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                 vKeysCell,10,true);
        }
        else if(!bRight && j >= 16){
            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                 vKeysCell,10,true);
        }
        else{
            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                 vKeysCell,iniThFAST,true);
        }*/


        if (vKeysCell.empty()) {
          // 如果默认阈值20采集不到特征点,则换成阈值7采集特征点
          FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
               vKeysCell, minThFAST, true);
          LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << i + 1 << " 行, 第 " << j + 1 << " 列 第 " << level + 1
                    << " 层图像换成阈值7采集特征点";
          /*if(bRight && j <= 13){
              FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                   vKeysCell,5,true);
          }
          else if(!bRight && j >= 16){
              FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                   vKeysCell,5,true);
          }
          else{
              FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                   vKeysCell,minThFAST,true);
          }*/
        }

        if (!vKeysCell.empty()) {
          for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++) {
            (*vit).pt.x += j * wCell;
            (*vit).pt.y += i * hCell;
            vToDistributeKeys.push_back(*vit);
          }
        } else {
          LOG(INFO) << __PRETTY_FUNCTION__ << " 第 " << i + 1 << " 行, 第 " << j + 1 << " 列 第 " << level + 1
                    << " 层图像特征点为 NULL";
        }

      }
    }

    // 保存图像金字塔所有层的特征点
    vector<KeyPoint> &keypoints = allKeypoints[level];
    keypoints.reserve(nfeatures);

    // 利用四叉树算法平均化特征点, 对当前层图像
    keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                  minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

    // 显示当前层的图像和特征点
    DisplayImageAndKeypoints(mvImagePyramid[level], level, keypoints);

    const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

    // Add border to coordinates and scale information
    const int nkps = keypoints.size();
    for (int i = 0; i < nkps; i++) {
      keypoints[i].pt.x += minBorderX;
      keypoints[i].pt.y += minBorderY;
      keypoints[i].octave = level;
      keypoints[i].size = scaledPatchSize;
    }
  }

  // compute orientations
  // 然后计算这些特征点的方向信息，注意这里还是分层计算的
  for (int level = 0; level < nlevels; ++level)
    computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
  LOG(INFO) << __PRETTY_FUNCTION__ << " end";
}

void ORBextractor::DisplayImageAndKeypoints(const Mat &inMat, const int level, const vector<KeyPoint> &inKeyPoint) {
  Mat out_put_image;
  cv::drawKeypoints(inMat, inKeyPoint, out_put_image);
  string str = to_string(level + 1) + " level keypoints";
  imshow(str, out_put_image);
  waitKey(0);
}

vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,
                                                     const int &minX,
                                                     const int &maxX,
                                                     const int &minY,
                                                     const int &maxY,
                                                     const int &N,
                                                     const int &level) {
  LOG(INFO) << __PRETTY_FUNCTION__ << " start";
  // Compute how many initial nodes
  // 计算应该生成的初始节点个数，根节点的数量 nIni 是根据边界的宽高比值确定的，一般是1或者2
  // 四舍五入到最邻近的整数
  // TODO:bug: 如果宽高比小于0.5，nIni=0, 后面hx会报错
  const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

  // 一个初始的节点的x方向有多少个像素
  const float hX = static_cast<float>(maxX - minX) / nIni;

  list<ExtractorNode> lNodes;

  vector<ExtractorNode *> vpIniNodes;
  vpIniNodes.resize(nIni);

  for (int i = 0; i < nIni; i++) {
    ExtractorNode ni;
    // 注意这里和提取FAST角点区域相同，都是“半径扩充图像”，特征点坐标从0 开始
    ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
    ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
    ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
    ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
    // 设置提取器结点的特征点数量
    ni.vKeys.reserve(vToDistributeKeys.size());

#ifdef SHOW_DIVIDE_IMAGE
    Mat tmp = mvImagePyramid[level];
    tmp.col(hX * static_cast<float>(i)) = (0, 0, 0);
    tmp.col(hX * static_cast<float>(i + 1)) = (0, 0, 0);
    tmp.row(0) = (0, 0, 0);
    tmp.row(maxY - minY) = (0, 0, 0);
    imshow("DistributeOctTree", tmp);
    waitKey(0);
#endif

    // 将刚才生成的提取节点添加到列表中
    // 虽然这里的ni是局部变量，但是由于这里的push_back()是拷贝参数的内容到一个新的对象中然后再添加到列表中
    // 所以当本函数退出之后这里的内存不会成为“野指针”
    lNodes.push_back(ni);
    // 存储这个初始的提取器节点句柄
    vpIniNodes[i] = &lNodes.back();
  }

  // Associate points to childs
  // 将特征点分配给子提取器结点
  for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
    // 获取这个特征点对象
    const cv::KeyPoint &kp = vToDistributeKeys[i];
    // 按特征点的横轴位置，分配给属于那个图像区域的提取器节点（最初的提取器节点）
    vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
  }

  // 这个迭代器很关键啊，后面四叉树算法里面有个 lNodes.front().lit = lNodes.begin() 貌似和这个有关，但是看其他人注释好像又没有关系
  // TODO: 到底有什么作用？
  list<ExtractorNode>::iterator lit = lNodes.begin();

  // 遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
  // TODO: ? 这个步骤是必要的吗？感觉可以省略，通过判断nIni个数和vKeys.size() 就可以吧
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
  }// 遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点

  //结束标志位清空
  bool bFinish = false;

  //记录迭代次数，只是记录，并未起到作用
  int iteration = 0;

  //用于存储节点的vSize和句柄对
  //这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
  vector<pair<int, ExtractorNode *> > vSizeAndPointerToNode;
  //调整大小，这里的意思是一个初始化节点将“分裂”成为四个，当然实际上不会有那么多，这里多分配了一些只是预防万一
  vSizeAndPointerToNode.reserve(lNodes.size() * 4);

  // 利用4叉树方法对图像进行划分区域
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
      //如果提取器节点只有一个特征点
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
            // TODO:?这个访问用的句柄貌似并没有用到？
            // lNodes.front().lit 和前面的迭代的lit 不同，只是名字相同而已
            // lNodes.front().lit是node结构体里的一个指针用来记录节点的位置
            // 迭代的lit 是while循环里作者命名的遍历的指针名称
            // lNodes.front().lit 获取list中的第一个元素变量中，类型为 std::list<ExtractorNode> 链表的迭代器 iterator
            lNodes.front().lit = lNodes.begin();
          }
        }
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

#ifdef SHOW_DIVIDE_IMAGE
        Mat tmp = mvImagePyramid[level];
        tmp.col(n1.BL.x) = (0, 0, 0);
        tmp.col(n2.BL.x) = (0, 0, 0);
        tmp.col(n2.BR.x) = (0, 0, 0);
        tmp.row(n1.BL.y) = (0, 0, 0);
        tmp.row(n1.UL.y) = (0, 0, 0);
        tmp.row(n3.BL.y) = (0, 0, 0);
        drawKeypoints(tmp, n1.vKeys, tmp);
        drawKeypoints(tmp, n2.vKeys, tmp);
        drawKeypoints(tmp, n3.vKeys, tmp);
        drawKeypoints(tmp, n4.vKeys, tmp);
        imshow("DistributeOctTree", tmp);
        waitKey(0);
#endif

        lit = lNodes.erase(lit);
        continue;
      }//判断当前遍历到的节点中是否有超过一个的特征点
    }//遍历列表中的所有提取器节点

    // Finish if there are more nodes than required features
    // or all nodes contain just one point
    // 停止这个过程的条件有两个，满足其中一个即可：
    // 1、当前的节点数已经超过了要求的特征点数
    // 2、当前所有的节点中都只包含一个特征点
    if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize) {
      bFinish = true;
    }
      // 如果存在待扩展的结点，那么最终会产生总共 当前结点数 + 待扩展结点数×3 个结点
      // 因为一分四之后，会删除原来的主节点，所以乘以3
      // 如果最终结点数量大于特征点数量
    else if (((int) lNodes.size() + nToExpand * 3) > N) {

      // 这里循环的意义是为了表示，如果再分裂一次那么就有可能 lNodes.size() >= N 了
      // 这里想办法尽可能使其刚刚达到或者超过要求的特征点个数时就退出
      while (!bFinish) {

        //获取当前的list中的节点个数
        prevSize = lNodes.size();

        //Prev这里是应该是保留的意思吧，保留那些还可以分裂的节点的信息, 这里是深拷贝
        vector<pair<int, ExtractorNode *> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
        vSizeAndPointerToNode.clear();

        // 将当前所有结点按照特征点数量排序，默认升序
        // 优先分裂特征点多的节点，使得特征点密集的区域保留更少的特征点
        //! 注意这里的排序规则非常重要！会导致每次最后产生的特征点都不一样。建议使用 stable_sort
        sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
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

        // 这里理想中应该是一个for循环就能够达成结束条件了，但是作者想的可能是，
        // 有些子节点所在的区域会没有特征点，因此很有可能一次for循环之后
        // 的数目还是不能够满足要求，所以还是需要判断结束条件并且再来一次
        if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize)
          bFinish = true;

      }
    }/*如果存在待扩展的结点，那么最终会产生总共 当前结点数 + 待扩展结点数×3 个结点
     因为一分四之后，会删除原来的主节点，所以乘以3
     如果最终结点数量大于特征点数量*/
    //当本次分裂后达不到结束条件但是再进行一次完整的分裂之后就可以达到结束条件时
  }// 利用4叉树方法对图像进行划分区域

  // Retain the best point in each node
  // 保留每个区域响应值最大的一个兴趣点
  // 使用这个 vector 来存储我们感兴趣的特征点的过滤结果
  vector<cv::KeyPoint> vResultKeys;
  //调整大小为要提取的特征点数目
  vResultKeys.reserve(nfeatures);
  // 遍历这个节点列表
  for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++) {
    // 得到这个节点区域中的特征点容器句柄
    vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
    // 得到指向第一个特征点的指针，后面作为最大响应值对应的关键点
    cv::KeyPoint *pKP = &vNodeKeys[0];
    // 用第1个关键点响应值初始化最大响应值
    // response 代表该点强壮大小，即该点是特征点的程度
    float maxResponse = pKP->response;
    // 开始遍历这个节点区域中的特征点容器中的特征点，注意是从1开始哟，0已经用过了
    for (size_t k = 1; k < vNodeKeys.size(); k++) {
      if (vNodeKeys[k].response > maxResponse) {
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }

    // 将这个节点区域中的响应值最大的特征点加入最终结果容器
    // TODO:只挑选一个特征点？？
    vResultKeys.push_back(*pKP);
  }
  LOG(INFO) << __PRETTY_FUNCTION__ << " end";
  return vResultKeys;
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4) {
  LOG(INFO) << __PRETTY_FUNCTION__ << " start";
  const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
  const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

  //Define boundaries of childs
  n1.UL = UL;
  n1.UR = cv::Point2i(UL.x + halfX, UL.y);
  n1.BL = cv::Point2i(UL.x, UL.y + halfY);
  n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
  n1.vKeys.reserve(vKeys.size());

  n2.UL = n1.UR;
  n2.UR = UR;
  n2.BL = n1.BR;
  n2.BR = cv::Point2i(UR.x, UL.y + halfY);
  n2.vKeys.reserve(vKeys.size());

  n3.UL = n1.BL;
  n3.UR = n1.BR;
  n3.BL = BL;
  n3.BR = cv::Point2i(n1.BR.x, BL.y);
  n3.vKeys.reserve(vKeys.size());

  n4.UL = n3.UR;
  n4.UR = n2.BR;
  n4.BL = n3.BR;
  n4.BR = BR;
  n4.vKeys.reserve(vKeys.size());

  //Associate points to childs
  for (size_t i = 0; i < vKeys.size(); i++) {
    const cv::KeyPoint &kp = vKeys[i];
    if (kp.pt.x < n1.UR.x) {
      if (kp.pt.y < n1.BR.y)
        n1.vKeys.push_back(kp);
      else
        n3.vKeys.push_back(kp);
    } else if (kp.pt.y < n1.BR.y)
      n2.vKeys.push_back(kp);
    else
      n4.vKeys.push_back(kp);
  }

  if (n1.vKeys.size() == 1)
    n1.bNoMore = true;
  if (n2.vKeys.size() == 1)
    n2.bNoMore = true;
  if (n3.vKeys.size() == 1)
    n3.bNoMore = true;
  if (n4.vKeys.size() == 1)
    n4.bNoMore = true;
  LOG(INFO) << __PRETTY_FUNCTION__ << " end";
}

const float factorPI = (float) (CV_PI / 180.f);
static void computeOrbDescriptor(const KeyPoint &kpt,
                                 const Mat &img, const Point *pattern,
                                 uchar *desc) {
  float angle = (float) kpt.angle * factorPI;
  float a = (float) cos(angle), b = (float) sin(angle);

  const uchar *center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
  const int step = (int) img.step;

#define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

  for (int i = 0; i < 32; ++i, pattern += 16) {
    int t0, t1, val;
    t0 = GET_VALUE(0);
    t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2);
    t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[i] = (uchar) val;
  }

#undef GET_VALUE
}

static void computeDescriptors(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors,
                               const vector<Point> &pattern) {
  descriptors = Mat::zeros((int) keypoints.size(), 32, CV_8UC1);

  for (size_t i = 0; i < keypoints.size(); i++)
    computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int) i));
}

int ORBextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints,
                             OutputArray _descriptors, std::vector<int> &vLappingArea) {
  //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
  if (_image.empty())
    return -1;

  Mat image = _image.getMat();
  assert(image.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  // 计算图像金字塔
  ComputePyramid(image);

  // 利用四叉树算法平均化特征点
  vector<vector<KeyPoint> > allKeypoints;
  ComputeKeyPointsOctTree(allKeypoints);
  //ComputeKeyPointsOld(allKeypoints);

  Mat descriptors;

  int nkeypoints = 0;
  for (int level = 0; level < nlevels; ++level)
    nkeypoints += (int) allKeypoints[level].size();
  if (nkeypoints == 0)
    _descriptors.release();
  else {
    _descriptors.create(nkeypoints, 32, CV_8U);
    descriptors = _descriptors.getMat();
  }

  //_keypoints.clear();
  //_keypoints.reserve(nkeypoints);
  _keypoints = vector<cv::KeyPoint>(nkeypoints);

  int offset = 0;
  //Modified for speeding up stereo fisheye matching
  int monoIndex = 0, stereoIndex = nkeypoints - 1;
  for (int level = 0; level < nlevels; ++level) {
    vector<KeyPoint> &keypoints = allKeypoints[level];
    int nkeypointsLevel = (int) keypoints.size();

    if (nkeypointsLevel == 0)
      continue;

    // preprocess the resized image
    Mat workingMat = mvImagePyramid[level].clone();
    GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

    // Compute the descriptors
    //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
    computeDescriptors(workingMat, keypoints, desc, pattern);

    offset += nkeypointsLevel;

    float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
    int i = 0;
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
             keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {

      // Scale keypoint coordinates
      if (level != 0) {
        keypoint->pt *= scale;
      }

      if (keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]) {
        _keypoints.at(stereoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(stereoIndex));
        stereoIndex--;
      } else {
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

