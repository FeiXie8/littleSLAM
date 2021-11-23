/**
 *路标类，每一个路标由多个特征点组成，这些特征点来自于不同位姿相机的观测
 **/

#pragma once
#include "myslam/common_include.h"

namespace myslam{
struct Frame;
struct Feature;

struct MapPoint{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_=0;
    bool is_outlierr_=false;
    Vec3 pos_=Vec3::Zero();
    std::mutex data_mutex_;
    int observed_times_=0;
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint(){};
    MapPoint(long id,Vec3 position);
    Vec3 Pos(){
        std::lock_guard<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void SetPos(const Vec3 &pos){
        std::lock_guard<std::mutex> lck(data_mutex_);
        pos_=pos;
    }

    void AddObservation(std::shared_ptr<Feature> feature){
        std::lock_guard<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }
    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    //工厂模式
    static MapPoint::Ptr CreateNewMappoint();
};
}