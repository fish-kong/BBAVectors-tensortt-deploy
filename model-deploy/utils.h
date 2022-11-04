#pragma once
#include <algorithm> 
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric> // std::iota 

using  namespace cv;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
struct alignas(float) Detection {
	//center_x center_y w h
	float bbox[8];
	float conf;  // bbox_conf * cls_conf
	int class_id;
};
static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0);
	float r_h = input_h / (img.rows*1.0);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	return out;
}
cv::Rect get_rect(cv::Mat& img, float bbox[4], int INPUT_W, int INPUT_H) {
	int l, r, t, b;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
	if (r_h > r_w) {
		//l = bbox[0] - bbox[2] / 2.f;
		//r = bbox[0] + bbox[2] / 2.f;

		//t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		//b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		//l = l / r_w;
		//r = r / r_w;
		//t = t / r_w;
		//b = b / r_w;

		l = bbox[0];
		r = bbox[2];
		t = bbox[1]- (INPUT_H - r_w * img.rows) / 2;
		b = bbox[3] - (INPUT_H - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;

	}
	return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
		(std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
		(std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
		(std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
		(std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}




void process_top_k(std::vector<float> scores, int top_K, std::vector<float> &scores_K, std::vector<int> &index_K) {


	std::vector<int> idx(scores.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(),
		[&scores](int index_1, int index_2) { return scores[index_1] > scores[index_2]; });
	// ªÒ»°K÷µ
	int k_num = std::min<int>(scores.size(), top_K);
	int idx_j = 0;
	for (int j = 0; j < k_num; ++j) {
		idx_j = idx[j];
		index_K.push_back(idx_j);
		scores_K.push_back(scores[idx_j]);
	}
}