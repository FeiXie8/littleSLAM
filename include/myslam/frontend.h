#pragma once
#include <opencv2/features2d.hpp>
#include"myslam/common_include.h"
#include"myslam/frame.h"
#include "myslam/map.h"

namespace myslam{
class  Backend;
class Viewer;
enum class FrontendStatus{INITING,TRACKING_GOOD,TRACKING_BAD,LOST};

class Frontend{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;
    Frontend();
    bool addFrame(Frame::Ptr frame);

    void setMap(Map::Ptr map){map_=map;}

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

    FrontendStatus GetStatus() const { return status_; }

    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        camera_left_ = left;
        camera_right_ = right;
    }


private:
    bool Track();
    bool Reset();
    int trackLastFrame();
    int estimateCurrentPose();
    bool insertKeyFrame();
    bool stereoInit();
    /**
     * 更新当前帧的特征点信息
     */ 
    int detectFeatures();
    int findFeaturesInRight();
    bool buildInitMap();
    int triangulateNewPoints();
    /**
     * 更新地图已有的目标对应的特征点
     */ 
    void setObservationsForKeyFrame();

    FrontendStatus status_=FrontendStatus::INITING;

    Frame::Ptr current_frame_=nullptr;
    Frame::Ptr last_frame_=nullptr;
    Camera::Ptr camera_left_=nullptr;
    Camera::Ptr camera_right_=nullptr;

    Map::Ptr map_=nullptr;
    std::shared_ptr<Backend> backend_=nullptr;
    std::shared_ptr<Viewer> viewer_=nullptr;

    SE3 relative_motion_;

    int tracking_inliers_=0; //当前帧跟踪到的特征点数量

    int num_features_=200;
    int num_features_init_=100;
    int num_features_tracking_=50;
    int num_features_tracking_bad=20;
    int num_features_needed_for_keyframe_=80;

    cv::Ptr<cv::GFTTDetector> gftt_;
};
}