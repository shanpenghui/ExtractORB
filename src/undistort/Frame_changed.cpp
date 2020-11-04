//
// Created by sph on 2020/11/3.
//
#include "Frame_changed.h"

void
UndistortKeyPoints(cv::Mat &mDistCoef, std::vector<cv::KeyPoint> &mvKeys,
                   std::vector<cv::KeyPoint> mvKeysUn, int N,
                   ORB_SLAM3::GeometricCamera *mpCamera, cv::Mat mK) {
    if (mDistCoef.at<float>(0) == 0.0) {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);

    for (int i = 0; i < N; i++) {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, static_cast<ORB_SLAM3::Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++) {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }

}

void ComputeImageBounds(const cv::Mat &imLeft, cv::Mat &mDistCoef,
                        ORB_SLAM3::GeometricCamera *mpCamera, cv::Mat &mK,
                float &mnMinX, float &mnMaxX, float &mnMinY, float &mnMaxY)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<ORB_SLAM3::Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = std::min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = std::max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = std::min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = std::max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void AssignFeaturesToGrid(int N, std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                          int Nleft, std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS],
                          std::vector<cv::KeyPoint> mvKeysUn,
                          std::vector<cv::KeyPoint> mvKeys,
                          std::vector<cv::KeyPoint> mvKeysRight,
                          float mnMinX, float mnMinY,
                          float mfGridElementWidthInv, float mfGridElementHeightInv
                          )
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                               : (i < Nleft) ? mvKeys[i]
                                                             : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY, mnMinX, mnMinY, mfGridElementWidthInv, mfGridElementHeightInv)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY, float mnMinX, float mnMinY,
               float mfGridElementWidthInv, float mfGridElementHeightInv)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

