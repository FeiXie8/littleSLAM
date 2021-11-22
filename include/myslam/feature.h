#pragma once
#include<memory>
#include<opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam{
struct Frame;
struct MapPoint;

//每一帧都对应一系列特征点，Feature类就代表这些特征点,注意和路标点类mappoint区分
struct Feature{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;  //TODO为什么要用weak_ptr避免循环引用，可以看一看成员变量是否包含了Feature类，这里是Frame持有了Feature
    
    cv::KeyPoint position_;
    std::weak_ptr<MapPoint> map_point_;  //?
    bool is_outlier_=false; //TODO是否为异常点，这里异常指什么？
    bool is_on_left_image_=true;
public:
    Feature(){}
    Feature(std::shared_ptr<Frame> frame,const cv::KeyPoint& kp):
            frame_(frame),position_(kp){}

};
}