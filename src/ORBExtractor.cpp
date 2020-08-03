//
// Created by sph on 2020/8/3.
//

#include "../inc/ORBExtractor.h"

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
//  umax.resize(HALF_PATCH_SIZE + 1);

//  int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
//  int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
//  const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
//  for (v = 0; v <= vmax; ++v)
//    umax[v] = cvRound(sqrt(hp2 - v * v));

  // Make sure we are symmetric
//  for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
//  {
//    while (umax[v0] == umax[v0 + 1])
//      ++v0;
//    umax[v] = v0;
//    ++v0;
//  }
  LOG(INFO) << __PRETTY_FUNCTION__ << "   end";
}

void ORBextractor::ComputePyramid(cv::Mat image) {
  for (int level = 0; level < nlevels; ++level) {
    float scale = mvInvScaleFactor[level];
    Size sz(cvRound((float) image.cols * scale), cvRound((float) image.rows * scale));
    Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
    Mat temp(wholeSize, image.type()), masktemp;
    mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

    // Compute the resized image
//    cout << "正在构建第 " << level + 1 << " 层金字塔" << endl;
    if (level != 0) {
      resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

      copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
//      cv::imshow("img_pyramid", mvImagePyramid[level]);
    } else {
      copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     cv::BORDER_REFLECT_101);
//      cv::imshow("img_pyramid", image);
    }
//    cv::waitKey(200);
  }

}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> > &allKeypoints) {
  LOG(INFO) << __PRETTY_FUNCTION__ << " start";

  allKeypoints.resize(nlevels);

  const float W = 30;

  for (int level = 0; level < nlevels; ++level) {
    const int minBorderX = EDGE_THRESHOLD - 3;
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
    const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

    vector<cv::KeyPoint> vToDistributeKeys;
    vToDistributeKeys.reserve(nfeatures * 10);

    const float width = (maxBorderX - minBorderX);
    const float height = (maxBorderY - minBorderY);

    const int nCols = width / W;
    const int nRows = height / W;
    const int wCell = ceil(width / nCols);
    const int hCell = ceil(height / nRows);

    for (int i = 0; i < nRows; i++) {
      const float iniY = minBorderY + i * hCell;
      float maxY = iniY + hCell + 6;

      if (iniY >= maxBorderY - 3)
        continue;
      if (maxY > maxBorderY)
        maxY = maxBorderY;

      for (int j = 0; j < nCols; j++) {
        const float iniX = minBorderX + j * wCell;
        float maxX = iniX + wCell + 6;
        if (iniX >= maxBorderX - 6)
          continue;
        if (maxX > maxBorderX)
          maxX = maxBorderX;

        vector<cv::KeyPoint> vKeysCell;

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
          FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
               vKeysCell, minThFAST, true);
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
        }

      }
    }

    vector<KeyPoint> &keypoints = allKeypoints[level];
    keypoints.reserve(nfeatures);

    keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                  minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

    cout << "正在构建第 " << level + 1 << " 层金字塔" << endl;
    Mat out_put_image;
    drawKeypoints(mvImagePyramid[level], keypoints, out_put_image);
    imshow("mvImagePyramid[level]'s keypoints", out_put_image);
    waitKey(0);

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
//  for (int level = 0; level < nlevels; ++level)
//    computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
  LOG(INFO) << __PRETTY_FUNCTION__ << " end";
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
  const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

  const float hX = static_cast<float>(maxX - minX) / nIni;

  list<ExtractorNode> lNodes;

  vector<ExtractorNode *> vpIniNodes;
  vpIniNodes.resize(nIni);

  for (int i = 0; i < nIni; i++) {
    ExtractorNode ni;
    ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
    ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
    ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
    ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
    ni.vKeys.reserve(vToDistributeKeys.size());

    lNodes.push_back(ni);
    vpIniNodes[i] = &lNodes.back();
  }

  //Associate points to childs
  for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
    const cv::KeyPoint &kp = vToDistributeKeys[i];
    vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
  }

  list<ExtractorNode>::iterator lit = lNodes.begin();

  while (lit != lNodes.end()) {
    if (lit->vKeys.size() == 1) {
      lit->bNoMore = true;
      lit++;
    } else if (lit->vKeys.empty())
      lit = lNodes.erase(lit);
    else
      lit++;
  }

  bool bFinish = false;

  int iteration = 0;

  vector<pair<int, ExtractorNode *> > vSizeAndPointerToNode;
  vSizeAndPointerToNode.reserve(lNodes.size() * 4);

  while (!bFinish) {
    iteration++;

    int prevSize = lNodes.size();

    lit = lNodes.begin();

    int nToExpand = 0;

    vSizeAndPointerToNode.clear();

    while (lit != lNodes.end()) {
      if (lit->bNoMore) {
        // If node only contains one point do not subdivide and continue
        lit++;
        continue;
      } else {
        // If more than one point, subdivide
        ExtractorNode n1, n2, n3, n4;
        lit->DivideNode(n1, n2, n3, n4);

        // Add childs if they contain points
        if (n1.vKeys.size() > 0) {
          lNodes.push_front(n1);
          if (n1.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
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

        lit = lNodes.erase(lit);
        continue;
      }
    }

    // Finish if there are more nodes than required features
    // or all nodes contain just one point
    if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize) {
      bFinish = true;
    } else if (((int) lNodes.size() + nToExpand * 3) > N) {

      while (!bFinish) {

        prevSize = lNodes.size();

        vector<pair<int, ExtractorNode *> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
        vSizeAndPointerToNode.clear();

        sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
        for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
          ExtractorNode n1, n2, n3, n4;
          vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

          // Add childs if they contain points
          if (n1.vKeys.size() > 0) {
            lNodes.push_front(n1);
            if (n1.vKeys.size() > 1) {
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

          lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

          if ((int) lNodes.size() >= N)
            break;
        }

        if ((int) lNodes.size() >= N || (int) lNodes.size() == prevSize)
          bFinish = true;

      }
    }
  }

  // Retain the best point in each node
  vector<cv::KeyPoint> vResultKeys;
  vResultKeys.reserve(nfeatures);
  for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++) {
    vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
    cv::KeyPoint *pKP = &vNodeKeys[0];
    float maxResponse = pKP->response;

    for (size_t k = 1; k < vNodeKeys.size(); k++) {
      if (vNodeKeys[k].response > maxResponse) {
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }

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

/*int ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                              OutputArray _descriptors, std::vector<int> &vLappingArea)
{
  LOG(INFO) << __PRETTY_FUNCTION__ << " start";
  //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
  if(_image.empty())
    return -1;

  Mat image = _image.getMat();
  assert(image.type() == CV_8UC1 );

  // Pre-compute the scale pyramid
  ComputePyramid(image);

  vector < vector<KeyPoint> > allKeypoints;
  ComputeKeyPointsOctTree(allKeypoints);
  //ComputeKeyPointsOld(allKeypoints);

  Mat descriptors;

  int nkeypoints = 0;
  for (int level = 0; level < nlevels; ++level)
    nkeypoints += (int)allKeypoints[level].size();
  if( nkeypoints == 0 )
    _descriptors.release();
  else
  {
    _descriptors.create(nkeypoints, 32, CV_8U);
    descriptors = _descriptors.getMat();
  }

  //_keypoints.clear();
  //_keypoints.reserve(nkeypoints);
  _keypoints = vector<cv::KeyPoint>(nkeypoints);

  int offset = 0;
  //Modified for speeding up stereo fisheye matching
  int monoIndex = 0, stereoIndex = nkeypoints-1;
  for (int level = 0; level < nlevels; ++level)
  {
    vector<KeyPoint>& keypoints = allKeypoints[level];
    int nkeypointsLevel = (int)keypoints.size();

    if(nkeypointsLevel==0)
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
             keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){

      // Scale keypoint coordinates
      if (level != 0){
        keypoint->pt *= scale;
      }

      if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
        _keypoints.at(stereoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(stereoIndex));
        stereoIndex--;
      }
      else{
        _keypoints.at(monoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(monoIndex));
        monoIndex++;
      }
      i++;
    }
  }

  //cout << "[ORBextractor]: extracted " << _keypoints.size() << " KeyPoints" << endl;
  LOG(INFO) << __PRETTY_FUNCTION__ << " end";
  return monoIndex;
}*/




