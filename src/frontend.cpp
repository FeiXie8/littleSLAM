//代码是当追踪到的特征点少了就会新增关键帧，只有当运动了较远时才会观测不到之前的路标点
//因此如果追踪效果好，此时就不做任何操作（甚至不估计当前位姿）？

#include<opencv2/opencv.hpp>
#include"myslam/algorithm.h"
#include"myslam/backend.h"
#include"myslam/config.h"
#include"myslam/feature.h"
#include"myslam/frontend.h"
#include"myslam/g2o_types.h"
#include"myslam/map.h"
#include"myslam/viewer.h"

namespace myslam{
Frontend::Frontend(){
    gftt_=cv::GFTTDetector::create(Config::Get<int>("num_features"),0.01,20);
    num_features_init_=Config::Get<int>("num_features_init");
    num_features_=Config::Get<int>("num_features");
}

bool Frontend::addFrame(myslam::Frame::Ptr frame){
    current_frame_=frame;

    switch(status_){
        case FrontendStatus::INITING:
            stereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_=current_frame_;
    return true;
}

/**
 * tracking_inliers是目前能观测到的路标点，如果较少，就将当前标记为bad，
 * 于是会在Addframe时新增关键帧，即新增路标点
 */
bool Frontend::Track(){
    if(last_frame_){
        current_frame_->setPose(relative_motion_*last_frame_->Pose());
    }

    int num_track_last=trackLastFrame();  //关键函数，从上一帧通过光流法得到当前帧特征点
    tracking_inliers_=estimateCurrentPose();

    if(tracking_inliers_>num_features_tracking_){  //判断跟踪情况好坏
        status_=FrontendStatus::TRACKING_GOOD;
    }else if(tracking_inliers_>num_features_tracking_bad){
        status_=FrontendStatus::TRACKING_BAD;
    }else{
        status_=FrontendStatus::LOST;
    }

    insertKeyFrame();
    relative_motion_=current_frame_->Pose()*last_frame_->Pose().inverse();

    if(viewer_) viewer_->addCurrentFrame(current_frame_);
    return true;
}

bool Frontend::insertKeyFrame(){
    if(tracking_inliers_>=num_features_needed_for_keyframe_){
        //目的是避免关键帧聚集在一起
        return false;
    }
    current_frame_->setKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO)<<"Set frame "<<current_frame_->id_<<"as keyframe "<<current_frame_->keyframe_id_;

    setObservationsForKeyFrame(); 

    detectFeatures(); //由于是关键帧，因此该帧的特征点不再只由光流得到，应该对该帧重新检测keypoints
    findFeaturesInRight();
    triangulateNewPoints();
    backend_->updateMap();

    if(viewer_) viewer_->updateMap();
    return true;
}

/**
 * 特征点通过LK追踪得到，该函数更新路标对应的最新像素点
 */
void Frontend::setObservationsForKeyFrame(){
    for(auto& feat : current_frame_->features_left_){
        auto mp=feat->map_point_.lock();
        if(mp) mp->AddObservation(feat);
    }
}

int Frontend::triangulateNewPoints(){
    std::vector<SE3> poses{camera_left_->pose(),camera_right_->pose()};
    SE3 current_pose_Twc=current_frame_->Pose().inverse();
    int cnt_triangulated_pts=0;
    for(size_t i=0;i<current_frame_->features_left_.size();++i){
        if(current_frame_->features_left_[i]->map_point_.expired() && 
            current_frame_->features_right_[i] != nullptr){
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                            current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                        current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld=Vec3::Zero();
            
            if(triangulation(poses,points,pworld) && pworld[2]>0){
                auto new_map_point=MapPoint::CreateNewMappoint();
                pworld=current_pose_Twc*pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]); //这里会添加一些nullptr

                current_frame_->features_left_[i]->map_point_=new_map_point;
                current_frame_->features_right_[i]->map_point_=new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO)<<"new landmarks: "<<cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::estimateCurrentPose(){
    typedef g2o::BlockSolver_6_3 BlockSolverType; //李代数与世界坐标
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType; //TODO这里的稠密求解有待进一步测试
    auto solver=new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    VertexPose* vertex_pose=new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    Mat33 K=camera_left_->K();

    int index=1;

    std::vector<EdgeProjectionPoseOnly*> edges;
    std::vector<Feature::Ptr> features;
    for(size_t i=0;i<current_frame_->features_left_.size();++i){
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }
    // 优化后如果误差很大，就将这些观测点设为坏点，对应的边将不加入后续优化
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // 将结果用来更新当前帧的位姿
    current_frame_->setPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::trackLastFrame() {
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {  //如果特征点有路标，就用路标重投影作为像素坐标
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {  //如果没有，就直接用像素坐标
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::stereoInit() {
    int num_features_left = detectFeatures();
    int num_coor_features = findFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = buildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->addCurrentFrame(current_frame_);
            viewer_->updateMap();
        }
        return true;
    }
    return false;
}

int Frontend::detectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}


int Frontend::findFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        auto mp = kp->map_point_.lock();
        //由于光流是基于优化的方法，这里是赋一个优化的初值
        if (mp) {
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::buildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            //这里的路标坐标为什么不乘上位姿，因为世界坐标系和初始时刻的相机坐标系位姿重合
            //双目相机的相机坐标系自己设置，不一定要与左相机重合
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->setKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->updateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}


}