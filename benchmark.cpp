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
        path = "../file_names.txt";
    }
    else if (argc == 2)
    {
        path = argv[1];
    }
    std::string param = "../model/scn.param";
    std::string bin = "../model/scn.bin";
    Classifier classifier(param, bin, false);
    
    if (path.substr(path.length() - 3) == "txt"){
        std::ifstream inFile;
        std::ifstream labels;
        labels.open("../labels.txt");
        inFile.open(path);
        if (!inFile) {
            std::cout << "Unable to open file\n";
            exit(1); // terminate with error
        }
        std::string imgPath;
        std::string s;
        std::string mapped[] = {"Neutral", "Happiness", "Sadness", "Anger"};
        int true_array[4] = {0, 0, 0, 0};
        int false_array[4] = {0, 0, 0, 0};
        int idx = 0;
        while (getline (inFile, imgPath) && getline (labels, s)) {
            idx++;
            int label = s[0] - '0';
            cv::Mat img = cv::imread(("../cropped_and_aligned/" + imgPath).c_str());
            auto t1 =cv::getTickCount();
            int out = classifier.Classify(img, 5);
            auto t2 =cv::getTickCount();
            if (label == out){
                true_array[label]++;
            }
            else{
                false_array[out]++;
            }
            std::cout<<"File index: "<<idx<<" | Forward time: "<<(t2-t1)/cv::getTickFrequency()*1000<<std::endl;

            // cv::putText(img, out, cv::Point(20, 20), cv::FONT_HERSHEY_DUPLEX, 1, CV_RGB(255, 0, 0), 1);
            // cv::imshow("Display window", img);
            // int k = cv::waitKey(0);
            //     if(k == 'q'){
            //         break;
            //     }
        }
        float acc = (true_array[0] + true_array[1] + true_array[2] + true_array[3])*1.0 / 
                    (false_array[0] + false_array[1] + false_array[2] + false_array[3] + 
                    true_array[0] + true_array[1] + true_array[2] + true_array[3]);
        float acc_per_class[4] = {0.0, 0.0, 0.0, 0.0};
        for (int i=0; i<4; i++){
            acc_per_class[i] = true_array[i]*1.0/(true_array[i] + false_array[i]);
        }

        std::cout<<"------------Summary Emotion on "<<9815<<" images-------------------"<<std::endl;
        std::cout<<"Accuracy all classes: "<<acc<<std::endl;
        std::cout<<"Neutral accuracy: "<<acc_per_class[0]<<std::endl;
        std::cout<<"Happiness accuracy: "<<acc_per_class[1]<<std::endl;
        std::cout<<"Sadness accuracy: "<<acc_per_class[2]<<std::endl;
        std::cout<<"Anger accuracy: "<<acc_per_class[3]<<std::endl;
        std::cout<<"------------End Summary Emotion on "<<9815<<" images-------------------"<<std::endl;

        inFile.close();
        labels.close();
    }
    else{
        cv::Mat img = cv::imread(path.c_str());
        int out = classifier.Classify(img, 5);
        std::cout<<out<<std::endl;
    }

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
