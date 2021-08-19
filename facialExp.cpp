#include "facialExp.h"

Classifier::Classifier():
        Net(new ncnn::Net())
{
}

inline void Classifier::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Classifier::Classifier(const std::string &model_param, const std::string &model_bin, bool stabling):
        Net(new ncnn::Net())
{
    Init(model_param, model_bin);
    SetDefaultParams();
    this->stabling = stabling;
}

void Classifier::Init(const std::string &model_param, const std::string &model_bin)
{
    Net->load_param(model_param.c_str());
    Net->load_model(model_bin.c_str());
}
inline void Classifier::SetDefaultParams(){
    _mean_val[0] = 0.485f*255.f;
    _mean_val[1] = 0.456f*255.f;
    _mean_val[2] = 0.406f*255.f;
    _std_val[0] = 1/0.229f/255.f;
    _std_val[1] = 1/0.224f/255.f;
    _std_val[2] = 1/0.225f/255.f;
}

Classifier::~Classifier(){
    Release();
}

void softmax(float *input, int len){
    int i;
    float m;
    /* Find maximum value from input array */
    m = input[0];
    for (i = 1; i < len; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    float sum = 0;
    for (i = 0; i < len; i++) {
        sum += expf(input[i]-m);
    }

    for (i = 0; i < len; i++) {
        input[i] = expf(input[i] - m - log(sum));
    }    
}

short get_result(std::queue<short> q){
    unsigned short value[] = {0, 0, 0, 0};
    std::queue<short> tmp = q; 
    while (!tmp.empty()) { 
        // std::cout<<tmp.front()<<" ";
        value[tmp.front()]++;
        tmp.pop(); 
    } 
    // std::cout<<"\n";
    short max_v = -1;
    short res = -1;
    for (short i=0; i<4; i++){
        if (value[i] > max_v){
            max_v = value[i];
            res = i;
        }
    }
    return res;
}

int Classifier::Classify(cv::Mat& bgr, unsigned short max_len){
    int w = bgr.cols;
    int h = bgr.rows;
    // scale
    // auto t1 =cv::getTickCount();

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, w, h, 112, 112);
    in.substract_mean_normalize(_mean_val, _std_val);
    // auto t2 =cv::getTickCount();
    // std::cout<<"process time: "<<(t2-t1)/cv::getTickFrequency()<<std::endl;
    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input_1", in);
    ncnn::Mat out;
    ex.extract("output_2", out);
    // auto t3 =cv::getTickCount();
    // std::cout<<"forward time: "<<(t3-t2)/cv::getTickFrequency()<<std::endl;
    float *score = out.channel(0);
    int pred = -1;
    float max = -999;
    for (short i=0; i<4; i++){
        if (score[i] > max){
            max = score[i];
            pred = i;
        }
    }
    softmax(score, 4);

    // for (short i=0; i<4; i++){
    //     std::cout<<score[i]<<std::endl;
    // }
    
    if (this->stabling == true){
        if (slice_window.size() < max_len){
                slice_window.push(pred);
            }
        else{
            slice_window.pop();
            slice_window.push(pred);
        }
        pred = get_result(slice_window);
    }

    return pred;

    // float class_score = score[pred];
    // if (class_score > 0.5){
    //     return label[pred];
    // }
    // return "\0";
}

cv::Mat FacePreprocess::meanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i ++)
    {
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++)
        {
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}

cv::Mat FacePreprocess::elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}

cv::Mat FacePreprocess::varAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return meanAxis0(temp_);
}

int FacePreprocess::MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}

cv::Mat FacePreprocess::similarTransform(cv::Mat src,cv::Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
	cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
	cv::SVD::compute(A, S,U, V);

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;

                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U* twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else{
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
            cv::Mat res = U* twp; // U
            T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
        }
        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d,S,res);
        float scale =  1.0/val*cv::sum(res).val[0];
        T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
        cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
        cv::Mat  temp2 = src_mean.t(); //src_mean.T
        cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
        cv::Mat temp4 = scale*temp3;
        T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
        T.rowRange(0, dim).colRange(0, dim) *= scale;
        return T;
    }

Detector::Detector(const std::string detect_face, const std::string detect_landmark){
    this->face_cascade.load(detect_face);
    facemark->loadModel(detect_landmark);
}

void Detector::detect(cv::Mat image, std::vector<cv::Rect> &faces, std::vector< std::vector<cv::Point2f> > &landmarks){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    face_cascade.detectMultiScale(gray, faces);
    facemark->fit(image, faces, landmarks);
}