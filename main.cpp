#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 参考 https://blog.csdn.net/fsFengQingYangheihei/article/details/73572856?locationNum=8&fps=1

using namespace std;
using namespace cv;

const int features_num = 1000;   // 最多提取的特征点的数量
const float scale_factor = 1.2f; // 金字塔图像之间的尺度参数
const int levels_num = 8;        // 金字塔层数
const int default_fast_T = 20;   // FAST默认检测阈值
const int min_fast_T = 7;        // FAST最小检测阈值
const int edge_threshold = 19;   // 边界尺度 对应ORB_SLAM3中的 const int EDGE_THRESHOLD = 19;
const int PATCH_SIZE = 31;

//定义四叉树节点类
class ExtractorNode {
 public:
  ExtractorNode() : bNoMore(false) {}

  void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

  std::vector<cv::KeyPoint> vkeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;   //用来判定是否继续分割

};

// 四叉树节点划分函数
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4) {
  /*四叉树四个节点示意图
 *
 * ---UL------------------------------------------------UR----------------
 *	/                          /                      /
 *	/                          /				      /
 *	/                          /				      /
 *	/                          /				      /
 *	/             n1           /		n2		      /
 *	/                          /				      /
 *	/                          /				      /
 *	/                          /				      /
 *	/                          / 				      /
 *	/--------------------------/------------------------------------
 *	/                          / 				      /
 *	/                          / 				      /
 *	/                          / 				      /
 *	/               n3         / 		n4		      /
 *	/                          / 				      /
 *	/                          / 				      /
 *	/                          / 				      /
 *	/                          / 				      /
 *	/                          /				      /
 *----BL------------------------------------------------BR-----------------
 *
 */

  const int halfx = ceil(static_cast<float>(UR.x - UL.x) / 2);
  const int halfy = ceil(static_cast<float>(BR.y - UL.y) / 2);

  n1.UL = UL;
  n1.UR = cv::Point2i(UL.x + halfx, UL.y);
  n1.BL = cv::Point2i(UL.x, UL.y + halfy);
  n1.BR = cv::Point2i(UL.x + halfx, UL.y + halfy);
  n1.vkeys.reserve(vkeys.size());

  n2.UL = n1.UR;
  n2.UR = UR;
  n2.BL = n1.BR;
  n2.BR = cv::Point2i(UR.x, UL.y + halfy);
  n2.vkeys.reserve(vkeys.size());

  n3.UL = n1.BL;
  n3.UR = n1.BR;
  n3.BL = BL;
  n3.BR = cv::Point2i(UL.x + halfx, BL.y);
  n3.vkeys.reserve(vkeys.size());

  n4.UL = n3.UR;
  n4.UR = n2.BR;
  n4.BL = n3.BR;
  n4.BR = BR;
  n4.vkeys.reserve(vkeys.size());

  for (size_t i = 0; i < vkeys.size(); ++i) {
    const cv::KeyPoint &kp = vkeys[i];
    if (kp.pt.x < n1.UR.x) {
      if (kp.pt.y < n1.BL.y)
        n1.vkeys.push_back(kp);
      else
        n3.vkeys.push_back(kp);
    } else if (kp.pt.y < n1.BR.y) {
      n2.vkeys.push_back(kp);
    } else {
      n4.vkeys.push_back(kp);
    }
  }

  if (n1.vkeys.size() == 1) {
    n1.bNoMore = true;
  }
  if (n2.vkeys.size() == 1) {
    n2.bNoMore = true;
  }
  if (n3.vkeys.size() == 1) {
    n3.bNoMore = true;
  }
  if (n4.vkeys.size() == 1) {
    n4.bNoMore = true;
  }

}

// 对应 ORBextractor.cc 中的函数 void ORBextractor::ComputePyramid(cv::Mat image)
//
void ComputePyramid(const int levels_num,
                    cv::Mat &img,
                    vector<float> &vec_scale_per_factor,
                    std::vector<cv::Mat> &vec_img_pyramid) {
  for (int level = 0; level < levels_num; ++level) {
    float scale = 1.0f / vec_scale_per_factor[level];
    cv::Size sz(cvRound((float) img.cols * scale), cvRound((float) img.rows * scale));

    if (level == 0) {
      vec_img_pyramid[level] = img;
    } else
      resize(vec_img_pyramid[level - 1], vec_img_pyramid[level], sz, 0, 0, CV_INTER_LINEAR);
    cout << "正在构建第 " << level + 1 << " 层金字塔" << endl;
    cv::imshow("img_pyramid", vec_img_pyramid[level]);
    cv::waitKey(100);
  }
}

int main() {
  // 1.读取图像
  cv::Mat img = cv::imread("../1520531124150444163.png", CV_LOAD_IMAGE_COLOR);
  // 1.读取图像 -合法性检查
  if (img.empty()) {
    cout << "no picture was found ...." << endl;
    return 0;
  } else
    cout << "img load successed!" << endl;

  // 2.构建图像金字塔 - 初始化数据结构
  vector<float> vec_scale_per_factor;     // 用于存储每层金字塔的尺度因子
  vec_scale_per_factor.resize(levels_num);// 金字塔一共8层, levels_num = 8
  vec_scale_per_factor[0] = 1.0f;         // 金字塔第一层的尺度是1, 即原图像
  for (int i = 1; i < levels_num; ++i) {  // 根据金字塔图像之间的尺度参数计算每层金字塔的尺度因子,存放在vec_scale_per_factor
    vec_scale_per_factor[i] = vec_scale_per_factor[i - 1] * scale_factor;
  }
  std::vector<cv::Mat> vec_img_pyramid(levels_num);// 用于存储每层金字塔的图像

  // 2.构建图像金字塔 - 使用cv::resize函数来构建
  ComputePyramid(levels_num, img, vec_scale_per_factor, vec_img_pyramid);
  cout << "*************************" << endl << endl;

  // 3.对金字塔的每层图像分析特征点 - 初始化数据结构
  vector<int> feature_num_per_level;  // TODO: 不知道什么含义
  std::vector<std::vector<cv::KeyPoint>> all_keypoints;
  all_keypoints.resize(levels_num);

  // 格子大小是30*30, TODO:这是什么格子？
  const float border_width = 30;

  // 3.对金字塔的每层图像分析特征点 - 循环每层图像
  for (int level = 0; level < levels_num; ++level) {
    // 计算每一层图像进行特征检测区域的边缘坐标,因为FAST是检测中心点附近3个点的范围,即半径是3
    const int min_boder_x = edge_threshold - 3;
    const int min_boder_y = min_boder_x;
    const int max_boder_x = vec_img_pyramid[level].cols - edge_threshold + 3;
    const int max_boder_y = vec_img_pyramid[level].rows - edge_threshold + 3;

    // 金字塔每层图像的特征点
    vector<cv::KeyPoint> vec_to_per_distribute_keys;
    vec_to_per_distribute_keys.reserve(features_num * 10);

    // 金字塔每层图像的宽/高/列/行
    const float width = max_boder_x - min_boder_x;
    const float height = max_boder_y - min_boder_y;
    const int cols = width / border_width;
    const int rows = height / border_width;
    // 金字塔每层图像格子宽和高
    const int width_cell = ceil(width / cols);
    const int height_cell = ceil(height / rows);
    cout << "第" << level + 1 << "层图像被切割成 " << rows << " 行 " << cols << " 列. 格子有 " << width_cell << " 列 " << height_cell
         << " 行 ";

    // 开始检测每个格子中的特征点，并统计图像中所有的特征点，为四叉树划分做准备
    for (int i = 0; i < rows; ++i) { // start of rows recycle
      // 行数计算Y值
      const float ini_y = min_boder_y + i * height_cell;
      float max_y = ini_y + height_cell + 6;
      // 处理非法Y值
      if (ini_y >= max_boder_y - 3)
        continue;
      if (max_y >= max_boder_y)
        max_y = max_boder_y;
      for (int j = 0; j < cols; ++j) { // start of cols recycle
        // 列数计算X值
        const float ini_x = min_boder_x + j * width_cell;
        float max_x = ini_x + width_cell + 6;
        // 处理非法X值
        if (ini_x >= max_boder_x - 6)
          continue;
        if (max_x >= max_boder_x)
          max_x = max_boder_x;

        std::vector<cv::KeyPoint> vec_keys_cell;

        // 对该层图像, 使用 OpenCV 自带的 FAST 角点检测函数, 默认检测阈值20
        cv::FAST(vec_img_pyramid[level].rowRange(ini_y, max_y).colRange(ini_x, max_x), vec_keys_cell,
                 default_fast_T,
                 true);

        // 如果默认检测阈值检测不到特征点, 则采用最小检测阈值7
        if (vec_keys_cell.empty()) {
          cv::FAST(vec_img_pyramid[level].rowRange(ini_y, max_y).colRange(ini_x, max_x), vec_keys_cell,
                   min_fast_T,
                   true);
        }

        if (!vec_keys_cell.empty()) {
          for (std::vector<cv::KeyPoint>::iterator vit = vec_keys_cell.begin();
               vit != vec_keys_cell.end(); vit++) {
            (*vit).pt.x += j * width_cell;
            (*vit).pt.y += i * height_cell;
            vec_to_per_distribute_keys.push_back(*vit);
          }

        }

      } // end of cols recycle

    } // end of rows recycle

    cout << "共有 " << vec_to_per_distribute_keys.size() << " 个特征点." << endl;

    // 开始划分四叉树节点的准备工作
    std::vector<cv::KeyPoint> &keypoints = all_keypoints[level];
    keypoints.reserve(features_num);

    // 划分根节点
    const int init_node_num = round(static_cast<float >(max_boder_x - min_boder_x) / (max_boder_y - min_boder_y));
    cout << "初始化时有 " << init_node_num << " 个根节点 ";

    const float interval_x = static_cast<float >(max_boder_x - min_boder_x) / init_node_num;
    cout << "节点间隔： " << interval_x << endl;

    std::list<ExtractorNode> list_nodes;   //用来存储所有的四叉树节点
    std::vector<ExtractorNode *> init_nodes_address;   //用来存储所有的四叉树节点中的vkeys
    init_nodes_address.resize(init_node_num);

    //根节点的UL，UR，BL，BR是不含边界的坐标，在最后的关键点坐边要补加上边界大小
    for (int i = 0; i < init_node_num; ++i) {
      ExtractorNode ni;
      ni.UL = cv::Point2i(interval_x * static_cast<float >(i), 0);
      ni.UR = cv::Point2i(interval_x * static_cast<float >(i + 1), 0);
      ni.BL = cv::Point2i(ni.UL.x, max_boder_y - min_boder_y);
      ni.BR = cv::Point2i(ni.UR.x, max_boder_y - min_boder_x);
      ni.vkeys.reserve(vec_to_per_distribute_keys.size());

      list_nodes.push_back(ni);
      init_nodes_address[i] = &list_nodes.back();
    }

    for (size_t i = 0; i < vec_to_per_distribute_keys.size(); ++i) {
      const cv::KeyPoint &kp = vec_to_per_distribute_keys[i];
      init_nodes_address[kp.pt.x / interval_x]->vkeys.push_back(kp);
    }

    //list_nodes中的结点和 init_nodes_address中的结点指针是同步的，只有在 list_nodes中存储的结点中存储了特征点，才能根据特征点的数目来决定如何处理这个结点
    //那如果在list_nodes中删除一个没有特征点的结点，那么在 init_nodes_address中对应的这个地址也会被销毁
    list<ExtractorNode>::iterator lit = list_nodes.begin();
    while (lit != list_nodes.end()) {
      if (lit->vkeys.size() == 1) {
        lit->bNoMore = true;
        lit++;
      } else if (lit->vkeys.empty()) {
        lit = list_nodes.erase(lit);
      } else
        lit++;
    }

    //开始划分四叉树节点
    bool is_finish = false;
    int iteration = 0;

    std::vector<std::pair<int, ExtractorNode *>> key_size_and_node;
    key_size_and_node.reserve(list_nodes.size() * 4);

    while (!is_finish) {
      iteration++;
      int pre_size = list_nodes.size();

      lit = list_nodes.begin();
      int to_expand_num = 0;
      key_size_and_node.clear();

      while (lit != list_nodes.end()) {
        if (lit->bNoMore) {
          lit++;
          continue;
        } else {
          ExtractorNode n1, n2, n3, n4;
          lit->DivideNode(n1, n2, n3, n4);

          if (n1.vkeys.size() > 0) {
            list_nodes.push_front(n1);
            if (n1.vkeys.size() > 1) {
              to_expand_num++;
              key_size_and_node.push_back(std::make_pair(n1.vkeys.size(), &list_nodes.front()));
              list_nodes.front().lit = list_nodes.begin();  //把节点中的迭代器指向自身，在后面的判断条件中使用
            }
          }
          if (n2.vkeys.size() > 0) {
            list_nodes.push_front(n2);
            if (n2.vkeys.size() > 1) {
              to_expand_num++;
              key_size_and_node.push_back(std::make_pair(n2.vkeys.size(), &list_nodes.front()));
              list_nodes.front().lit = list_nodes.begin();
            }
          }
          if (n3.vkeys.size() > 0) {
            list_nodes.push_front(n3);
            if (n3.vkeys.size() > 1) {
              to_expand_num++;
              key_size_and_node.push_back(std::make_pair(n3.vkeys.size(), &list_nodes.front()));
              list_nodes.front().lit = list_nodes.begin();
            }
          }
          if (n4.vkeys.size() > 0) {
            list_nodes.push_front(n4);
            if (n4.vkeys.size() > 1) {
              to_expand_num++;
              key_size_and_node.push_back(std::make_pair(n4.vkeys.size(), &list_nodes.front()));
              list_nodes.front().lit = list_nodes.begin();
            }
          }

          lit = list_nodes.erase(lit);
          continue;
        }

      }

      //计算当前level上应该分布多少特征点
      //等比数列sn = a1(1-q^level)/(1-q)
      //sn是总的特征数目，a1是level=0 层上的特征点数

      feature_num_per_level.resize(levels_num);
      float factor = 1.0f / scale_factor;
      float desired_feature_per_scale =
          features_num * (1 - factor) / (1 - (float) pow((double) factor, (double) levels_num));
      int sum_features = 0;
      for (int i = 0; i < levels_num - 1; ++i) {
        feature_num_per_level[i] = cvRound(desired_feature_per_scale);
        sum_features += feature_num_per_level[i];
        desired_feature_per_scale *= factor;
      }
      feature_num_per_level[levels_num - 1] = std::max(features_num - sum_features, 0);

      //判断四叉树结束条件
      //当创建的结点的数目比要求的特征点还要多或者，每个结点中都只有一个特征点的时候就停止分割
      if ((int) list_nodes.size() >= features_num || (int) list_nodes.size() == pre_size) {
        is_finish = true;

      } //如果现在生成的结点全部进行分割后生成的结点满足大于需求的特征点的数目，但是不继续分割又不能满足大于N的要求时
        //这里为什么是乘以三，这里也正好印证了上面所说的当一个结点被分割成四个新的结点时，
        //这个结点时要被删除的，其实总的结点时增加了三个
      else if (((int) list_nodes.size() + to_expand_num * 3) > feature_num_per_level[level]) {
        while (!is_finish) {
          pre_size = list_nodes.size();
          std::vector<std::pair<int, ExtractorNode *> > prve_size_and_node_adderss = key_size_and_node;
          key_size_and_node.clear();

          sort(prve_size_and_node_adderss.begin(), prve_size_and_node_adderss.end());

          for (int j = prve_size_and_node_adderss.size() - 1; j >= 0; --j) {
            ExtractorNode n1, n2, n3, n4;
            prve_size_and_node_adderss[j].second->DivideNode(n1, n2, n3, n4);

            if (n1.vkeys.size() > 0) {
              list_nodes.push_front(n1);
              if (n1.vkeys.size() > 1) {
                key_size_and_node.push_back(std::make_pair(n1.vkeys.size(), &list_nodes.front()));
                list_nodes.front().lit = list_nodes.begin();
              }
            }

            if (n2.vkeys.size() > 0) {
              list_nodes.push_front(n2);
              if (n2.vkeys.size() > 1) {
                key_size_and_node.push_back(std::make_pair(n2.vkeys.size(), &list_nodes.front()));
                list_nodes.front().lit = list_nodes.begin();
              }
            }
            if (n3.vkeys.size() > 0) {
              list_nodes.push_front(n3);
              if (n3.vkeys.size() > 1) {
                key_size_and_node.push_back(std::make_pair(n3.vkeys.size(), &list_nodes.front()));
                list_nodes.front().lit = list_nodes.begin();
              }
            }
            if (n4.vkeys.size() > 0) {
              list_nodes.push_front(n4);
              if (n4.vkeys.size() > 1) {
                key_size_and_node.push_back(std::make_pair(n4.vkeys.size(), &list_nodes.front()));
                list_nodes.front().lit = list_nodes.begin();
              }
            }

            list_nodes.erase(prve_size_and_node_adderss[j].second->lit);
            if ((int) list_nodes.size() >= feature_num_per_level[level]) {
              break;
            }
          }

          if ((int) list_nodes.size() >= features_num || (int) list_nodes.size() == pre_size)
            is_finish = true;
        }

      }
    }

    //四叉树划分完成，在每个节点中留下一个keypoints
    std::vector<cv::KeyPoint> result_keys;
    result_keys.reserve(feature_num_per_level[level]);

    for (std::list<ExtractorNode>::iterator lit = list_nodes.begin(); lit != list_nodes.end(); lit++) {
      vector<cv::KeyPoint> &node_keys = lit->vkeys;
      cv::KeyPoint *keypoint = &node_keys[0];
      float max_response = keypoint->response;

      for (size_t k = 1; k < node_keys.size(); ++k) {
        if (node_keys[k].response > max_response) {
          keypoint = &node_keys[k];
          max_response = node_keys[k].response;
        }
      }

      result_keys.push_back(*keypoint);

    }

    keypoints = result_keys;

    //添加边界尺寸，计算特征点Patch的大小，根据每层的尺度的不同而不同
    const int scale_patch_size = PATCH_SIZE * vec_scale_per_factor[level];

    const int kps = keypoints.size();
    for (int l = 0; l < kps; ++l) {
      keypoints[l].pt.x += min_boder_x;
      keypoints[l].pt.y += min_boder_y;
      keypoints[l].octave = level;           //特征点所在图像金字塔的层
      keypoints[l].size = scale_patch_size; //特征点邻域直径，因为最初的网格大小是30，所以乘上比例系数
    }
    cout << "经过四叉数筛选, 第 " << level + 1 << " 层剩余 " << result_keys.size() << " 个特征点" << endl;

  }

  //统计所有层的特征点并进行尺度恢复
  int num_keypoints = 0;
  for (int level = 0; level < levels_num; ++level) {
    num_keypoints += (int) all_keypoints[level].size();
  }
  cout << "total " << num_keypoints << " keypoints" << endl;

  vector<cv::KeyPoint> out_put_all_keypoints(num_keypoints);

  for (int level = 0; level < levels_num; ++level) {
    if (level == 0) {
      for (int i = 0; i < all_keypoints[level].size(); ++i) {
        out_put_all_keypoints.push_back(all_keypoints[level][i]);
      }
    }

    float scale = vec_scale_per_factor[level];
    for (vector<cv::KeyPoint>::iterator key = all_keypoints[level].begin();
         key != all_keypoints[level].end(); key++) {
      key->pt *= scale; //尺度恢复
    }

    out_put_all_keypoints.insert(out_put_all_keypoints.end(), all_keypoints[level].begin(),
                                 all_keypoints[level].end());
  }

  //对比试验
  cv::Mat out_img1;
  cv::drawKeypoints(img, out_put_all_keypoints, out_img1);
  cv::imshow("四叉数", out_img1);
  waitKey(0);

  cv::Mat img2;
  vector<cv::KeyPoint> fast_keypoints;
  Ptr<ORB> orb = ORB::create(1000);
  orb->detect(img, fast_keypoints);
  drawKeypoints(img, fast_keypoints, img2);
  cv::imshow("orb", img2);
  cv::waitKey(0);

  return 0;
}