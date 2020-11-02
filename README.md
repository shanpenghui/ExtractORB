# Feature_Extractor
To understand the principle of ORB-SLAM I create this subproject.
Contains two parts for now:
> Distribute Oct Tree
>
> Orb Extractor
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
