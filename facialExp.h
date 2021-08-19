#include <opencv2/opencv.hpp>
#include "net.h"
#include <string>
#include <queue>
#include <math.h>
#include <vector>


class Classifier{
    public:
        Classifier();
        void Init(const std::string &model_param, const std::string &model_bin);
        Classifier(const std::string &model_param, const std::string &model_bin, bool stabling = false);
        inline void Release();
        inline void SetDefaultParams();
        int Classify(cv::Mat& bgr, unsigned short max_len = 1);
        ~Classifier();
        ncnn::Net *Net;
        bool stabling;
        std::queue<short> slice_window;
        float _mean_val[3];
        float _std_val[3];
};

class Detector{
    public:
        Detector();
        Detector(const std::string model_name, const std::string detect_landmark);
        cv::CascadeClassifier face_cascade;
        cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
        void detect(cv::Mat image, std::vector<cv::Rect> &faces, std::vector< std::vector<cv::Point2f> > &landmarks);
};

namespace FacePreprocess {

    cv::Mat meanAxis0(const cv::Mat &src);
    cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B);
    cv::Mat varAxis0(const cv::Mat &src);
    int MatrixRank(cv::Mat M);
    cv::Mat similarTransform(cv::Mat src,cv::Mat dst);
}
