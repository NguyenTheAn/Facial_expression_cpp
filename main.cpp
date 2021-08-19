#include "facialExp.h"
#include <fstream>

int main(int argc, char** argv){
    float v1[5][2] = {
    {30.2946f + 8.0f, 51.6963f},
    {65.5318f + 8.0f, 51.5014f},
    {48.0252f + 8.0f, 71.7366f},
    {33.5493f + 8.0f, 92.3655f},
    {62.7299f + 8.0f, 92.2041f}};
    cv::Mat src(5,2,CV_32FC1, v1);

    std::string path;
    if  (argc == 1)
    {
        path = "../test/00001.png";
    }
    else if (argc == 2)
    {
        path = argv[1];
    }
    std::string param = "../model/scn.param";
    std::string bin = "../model/scn.bin";
    Detector detector("../model/haarcascade_frontalface_alt2.xml", "../model/lbfmodel.yaml");
    Classifier classifier(param, bin, true);

    cv::VideoCapture cam("../phuongpt94_happiness.avi");
    cv::Mat frame;
    std::string mapped[] = {"Neutral", "Happiness", "Sadness", "Anger"};
    while(cam.read(frame)){
        cv::Mat img = frame.clone();
        std::vector<cv::Rect> faces;
        std::vector< std::vector<cv::Point2f> > landmarks;
        detector.detect(frame, faces, landmarks);
        for (auto face : faces){
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0));
        }

        for(auto landmark : landmarks){
            float landmark_5_points[5][2] = {
                {((landmark[36] + landmark[39])/2).x, ((landmark[36] + landmark[39])/2).y},
                {((landmark[42] + landmark[45])/2).x, ((landmark[42] + landmark[45])/2).y},
                {landmark[30].x, landmark[30].y},
                {landmark[48].x, landmark[48].y},
                {landmark[54].x, landmark[54].y}};

            cv::Mat dst(5,2,CV_32FC1, landmark_5_points);
            cv::Mat m = FacePreprocess::similarTransform(dst, src);

            cv::Rect roi(0, 0, 3, 2);
            cv::Mat M = m(roi);
            cv::Mat warpImg;
            cv::warpAffine(img, warpImg, M, cv::Size(112, 112));
            auto t1 =cv::getTickCount();
            int pred = classifier.Classify(warpImg, 5);
            std::string out = mapped[pred];
            auto t2 =cv::getTickCount();
            std::cout<<"Forward time: "<<(t2-t1)/cv::getTickFrequency()*1000<<std::endl;
            cv::putText(frame, out, cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 0, 255), 1);

            cv::circle(frame, (landmark[36] + landmark[39])/2, 3, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(frame, (landmark[42] + landmark[45])/2, 3, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(frame, landmark[30], 3, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(frame, landmark[48], 3, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(frame, landmark[54], 3, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 'q') {
            cv::destroyAllWindows();
            break;
        }
    }
    
    // if (path.substr(path.length() - 3) == "txt"){
    //     std::ifstream inFile;
    //     inFile.open(path);
    //     if (!inFile) {
    //         std::cout << "Unable to open file\n";
    //         exit(1); // terminate with error
    //     }
    //     std::string imgPath;
    //     while (getline (inFile, imgPath)) {
    //         cv::Mat img = cv::imread(("../" + imgPath).c_str());
    //         std::string out = classifier.Classify(img, 5);
    //         cv::putText(img, out, cv::Point(10, 10), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(118, 185, 0), 1);
    //         cv::imshow("Display window", img);
    //         int k = cv::waitKey(0);
    //             if(k == 'q'){
    //                 break;
    //             }
    //     }

    //     inFile.close();
    // }
    // else{
    //     cv::Mat img = cv::imread(path.c_str());
    //     std::string out = classifier.Classify(img, 5);
    //     if (out != "\0")
        
    //         std::cout<<out<<std::endl;
    //     else{
    //         std::cout<<"None"<<std::endl;
    //     }
    // }

    // cv::Mat img = cv::imread(path.c_str());
    // auto t1 =cv::getTickCount();
    // std::string out = classifier.Classify(img, 5);
    // auto t2 =cv::getTickCount();
    // std::cout<<"process time: "<<(t2-t1)/cv::getTickFrequency()<<std::endl;
    // if (out != "\0")
    
    //     std::cout<<out<<std::endl;
    // else{
    //     std::cout<<"None"<<std::endl;
    // }
    // auto t3 =cv::getTickCount();
    // float v1[5][2] = {
    // {30.2946f, 51.6963f},
    // {65.5318f, 51.5014f},
    // {48.0252f, 71.7366f},
    // {33.5493f, 92.3655f},
    // {62.7299f, 92.2041f}};

    // cv::Mat src(5,2,CV_32FC1, v1);

    // memcpy(src.data, v1, 2 * 5 * sizeof(float));

    // float v2[5][2] = 
    // {{932.21136, 301.58768},
    // {1174.639, 316.11005},
    // {1091.5206, 391.77417},
    // {950.5747, 566.9194},
    // {1149.6477, 577.0574}};

    // cv::Mat dst(5,2,CV_32FC1, v2);

    // memcpy(dst.data, v2, 2 * 5 * sizeof(float));

    // cv::Mat m = FacePreprocess::similarTransform(dst, src);

    // cv::Rect roi(0, 0, 3, 2);
    // cv::Mat M = m(roi);
    // cv::Mat warpImg;
    // cv::warpAffine(img, warpImg, M, cv::Size(112, 112));
    // cv::imshow("Display window", warpImg);
    // cv::waitKey(0);
    // std::cout<<"process time: "<<(t3-t2)/cv::getTickFrequency()<<std::endl;
    
    return 0;
}
