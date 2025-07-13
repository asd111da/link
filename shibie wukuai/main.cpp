#include <opencv2/opencv.hpp>
#include <iostream>

int thres[6] = {35, 85, 50, 255, 50, 255};

int main()
{
    cv::Mat imgs, backproj;
    cv::VideoCapture video("../data/rgb.avi");
    if (!video.isOpened()) {
        std::cerr << "无法打开视频" << std::endl;
        return -1;
    }

    // 创建调节窗口
    cv::namedWindow("Control", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("LowH", "Control", &thres[0], 179);
    cv::createTrackbar("HighH", "Control", &thres[1], 179);
    cv::createTrackbar("LowS", "Control", &thres[2], 255);
    cv::createTrackbar("HighS", "Control", &thres[3], 255);
    cv::createTrackbar("LowV", "Control", &thres[4], 255);
    cv::createTrackbar("HighV", "Control", &thres[5], 255);

    bool trackInit = false;
    cv::Rect greenBox;
    cv::Mat hist;

    int hsize = 16;
    float hranges[] = {0, 180};
    const float* phranges = hranges;

    while (true)
    {
        video >> imgs;
        if (imgs.empty()) break;

        // 1. HSV分离
        cv::Mat hsv;
        cv::cvtColor(imgs, hsv, cv::COLOR_BGR2HSV);

        // 提取颜色掩膜
        cv::Mat maskR, maskG, maskB;
        cv::inRange(hsv, cv::Scalar(0, 70, 30), cv::Scalar(10, 255, 255), maskR);
        cv::inRange(hsv, cv::Scalar(thres[0], thres[2], thres[4]),
                          cv::Scalar(thres[1], thres[3], thres[5]), maskG);
        cv::inRange(hsv, cv::Scalar(100, 100, 30), cv::Scalar(130, 255, 255), maskB);

        // 形态学操作
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
        for (auto& mask : {maskR, maskG, maskB}) {
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        }

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contoursR, contoursG, contoursB;
        std::vector<cv::Vec4i> hier;
        cv::findContours(maskR, contoursR, hier, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(maskG, contoursG, hier, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(maskB, contoursB, hier, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat img_result = imgs.clone();

        // 红色标记
        for (const auto& cnt : contoursR) {
            double area = cv::contourArea(cnt);
            if (area < 1000 || area > 30000) continue;
            cv::Rect box = cv::boundingRect(cnt);
            cv::rectangle(img_result, box, cv::Scalar(0, 0, 255), 2);
            cv::putText(img_result, "Red", box.tl() - cv::Point(0, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        // 蓝色标记
        for (const auto& cnt : contoursB) {
            double area = cv::contourArea(cnt);
            if (area < 1000 || area > 30000) continue;
            cv::Rect box = cv::boundingRect(cnt);
            cv::rectangle(img_result, box, cv::Scalar(255, 0, 0), 2);
            cv::putText(img_result, "Blue", box.tl() - cv::Point(0, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        }

        // 跟踪绿色目标
        if (!trackInit && !contoursG.empty()) {
            // 取最大绿色轮廓初始化跟踪区域
            double maxArea = 0;
            for (const auto& cnt : contoursG) {
                double area = cv::contourArea(cnt);
                if (area > maxArea && area < 30000) {
                    greenBox = cv::boundingRect(cnt);
                    maxArea = area;
                }
            }

            if (greenBox.area() > 0) {
                // 提取 H 通道
                cv::Mat hue;
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

                cv::Mat roi(hue, greenBox);
                cv::calcHist(&roi, 1, 0, cv::Mat(), hist, 1, &hsize, &phranges);
                cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
                trackInit = true;
            }
        }

        if (trackInit) {
            // 继续跟踪绿色目标
            cv::Mat hue;
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

            cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            cv::meanShift(backproj, greenBox,
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

            // 显示绿色跟踪框
            cv::rectangle(img_result, greenBox, cv::Scalar(0, 255, 0), 2);
            cv::putText(img_result, "Green", greenBox.tl() - cv::Point(0, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // 显示所有结果
        cv::imshow("颜色检测结果", img_result);

        int key = cv::waitKey(30);
        if (key == 27) break;
        else if (key == 32) {
            while (true) {
                int pauseKey = cv::waitKey(0);
                if (pauseKey == 32 || pauseKey == 27) break;
            }
        }
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}
