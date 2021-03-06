#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam{
Viewer::Viewer(){
    viewer_thread_=std::thread(std::bind(&Viewer::threadLoop,this));
}

void Viewer::close(){
    viewer_running_=false;
    viewer_thread_.join();
}

void Viewer::addCurrentFrame(Frame::Ptr current_frame){
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    current_frame_=current_frame;
}

void Viewer::updateMap(){
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    assert(map_!=nullptr);
    active_keyframes_=map_->GetActiceKeyFrames();
    active_landmarks_=map_->GetActiveMapPoints();
    map_update_=true;
}

void Viewer::threadLoop(){
    pangolin::CreateWindowAndBind("Myslam",1024,768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    //摆放相机
    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),//TODO这里是相机的参数fx,fy,cx,cy,最短视距，最长视距
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

    //交互窗口
    pangolin::View& vis_display=
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3]={0,0,1};
    const float green[3]={0,1,0};

    while(!pangolin::ShouldQuit()&&viewer_running_){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        vis_display.Activate(vis_camera);

        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        if(current_frame_){
            drawFrame(current_frame_,green);
            followCurrentFrame(vis_camera);

            cv::Mat img=plotFrameImage();
            cv::imshow("image",img);
            cv::waitKey(1);
        }

        if(map_){
            drawMapPoints();
        }

        pangolin::FinishFrame();
        usleep(5000);
    }
    LOG(INFO)<<"STOP viewer";
}

cv::Mat Viewer::plotFrameImage(){
    cv::Mat img_out;
    cv::cvtColor(current_frame_->left_img_,img_out,cv::COLOR_GRAY2BGR);
    for(size_t i=0;i<current_frame_->features_left_.size();i++){
        if(current_frame_->features_left_[i]->map_point_.lock()){
            auto feat=current_frame_->features_left_[i];
            cv::circle(img_out,feat->position_.pt,2,cv::Scalar(0,250,0),2);
        }
    }
    return img_out;
}


void Viewer::followCurrentFrame(pangolin::OpenGlRenderState& vis_camera){
    SE3 Twc=current_frame_->Pose().inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m,true);
}

/**
 * 这个函数是Opengl常用思想，每一次变换都是从相对于世界坐标系0位姿开始的，这也符合编程逻辑 
 */
void Viewer::drawFrame(Frame::Ptr frame,const float* color){
    SE3 Twc=frame->Pose().inverse();
    const float sz=1.0;
    const int line_width=2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();

    Sophus::Matrix4f m=Twc.matrix().template cast<float>(); //TODO这里实验一下template的作用
    glMultMatrixf((GLfloat*)m.data());
    
    if(color==nullptr){
        glColor3f(1,0,0);
    }else{
        glColor3f(color[0],color[1],color[2]);
    }

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::drawMapPoints(){
    const float red[3]={1.0,0,0};
    for(auto& kf:active_keyframes_){
        drawFrame(kf.second,red);
    }
    glPointSize(2);
    glBegin(GL_POINTS);
    for(auto& landmark:active_landmarks_){
        auto pos=landmark.second->Pos();
        glColor3f(red[0],red[1],red[2]);
        glVertex3d(pos[0],pos[1],pos[2]);
    }
}
}