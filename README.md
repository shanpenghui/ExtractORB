# ExtractORB
This project is to show the process of ExtractORB function in ORB-SLAM3 which is in file ORBextractor.cc.
>void Frame::ExtractORB(int flag, const cv::Mat &im)
>

Help to understand the ORB-SLAM3 step by step.

I pull functions related as a class named ORBextractor, which contains the functions below:
> ORBextractor
>
> ComputePyramid
>
>IC_Angle
>
>computeOrientation
>
>ComputeKeyPointsOctTree
>
>DisplayImageAndKeypoints
>
>DistributeOctTree
>
>DivideNode
>
>computeOrbDescriptor
>
>computeDescriptors
>
>operator()
>


## Usage
1.Install
Google Log and OpenCV is needed.
```shell script
git clone https://github.com/google/glog
cd glog
cmake -H. -Bbuild -G "Unix Makefiles"
cmake --build build
cmake --build build --target test
cd build
sudo make install
```

2.Build
```shell script
mkdir build
cd build
cmake ..
make -j 4
```

3.result
![](https://github.com/shanpenghui/ExtractORB/tree/master/img_folder/Screenshot.png)