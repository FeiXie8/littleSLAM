#pragma once

#include "myslam/camera.h"
#include "myslam/common_include.h"
namespace myslam{
struct MapPoint;
struct Feature;

//每一帧都有一个自己的id，关键帧有关键帧id
struct Frame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long id_;
    unsigned long keyframe_id_=0;
    bool is_keyframe_=false;
    double time_stamp_;
    SE3 pose_;
    std::mutex pose_mutex_;
    cv::Mat left_img_,right_img_;

    std::vector<std::shared_ptr<Feature>> features_left_;
    std::vector<std::shared_ptr<Feature>> features_right_;
public:
    Frame(){}
    Frame(long id,double time_stamp,const SE3& pose,const Mat& left,const Mat& right);
    SE3 Pose(){
        std::lock_guard<std::mutex> lck(pose_mutex_);
        return pose_;
    }
    void setKeyFrame();
    static std::shared_ptr<Frame> CreateFrame();//工厂构建模式，分配id
};

}//namespace myslam