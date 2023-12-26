#include "aruco_nano.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fmt/core.h>
#include <opencv2/imgproc.hpp>

using namespace aruconano;
using namespace std::chrono;
using namespace std::chrono_literals;

using double_ms_dur = std::chrono::duration<double, std::milli>;

static cv::Mat image_1829px, image_1829px_gray;
static cv::Mat image_3658px, image_3658px_gray;
static constexpr int kWarmupLoops = 10;

std::pair<double_ms_dur, std::vector<Marker>> aruco_v5_detect(const cv::Mat &inputMat) {
	std::vector<Marker> detectedMarkers;

	steady_clock::time_point begin = steady_clock::now();
	auto markers = MarkerDetector::detect(inputMat, 10, TagDicts::APRILTAG_36h11);
	double_ms_dur detectElapsed = steady_clock::now() - begin;

	return {detectElapsed, detectedMarkers};
}


void runTestSuite(bool parallel, bool hires, std::function<double(const cv::Mat&)> detectFunc, std::string detectName, const int iters = 500) {
	cv::Mat colorMat = hires ? image_3658px : image_1829px;
	cv::Mat grayMat = hires ? image_3658px_gray : image_1829px_gray;

	fmt::println("[{}] Loaded image: {} x {}. Paralellization: {}", 
		detectName, colorMat.cols, colorMat.rows, (parallel ? "ON" : "OFF"));
	fmt::println("[{}] Running {} iterations for color and grayscale", detectName, iters);

	cv::setNumThreads(parallel ? -1 : 0);

	const int iterations = 500;
	double colorMeans = 0.0;
	double grayMeans = 0.0;

	for (int i = 0; i < iterations; i++) {
		double time = detectFunc(colorMat);
		colorMeans += time;
	}
	for (int i = 0; i < iterations; i++) {
		double time = detectFunc(grayMat);
		grayMeans += time;
	}

	double colorMean = colorMeans / (double)iterations;
	double grayMean = grayMeans / (double)iterations;
	double grayDiff = colorMean - grayMean;
	double grayDiffPct = (grayDiff / colorMean) * 100.0;

	fmt::println("[{}] Mean (Color): {:.4f}ms", detectName, colorMean);
	fmt::println("[{}] Mean (Gray): {:.4f}ms", detectName, grayMean);
	fmt::println("[{}] Perf lift: {:.4f}ms ({:.2f}%)", detectName, grayDiff, grayDiffPct);
}

struct TestEntry {
	std::string name;
	bool hires;
	bool parallel;
	std::function<double(const cv::Mat&)> func;
};

int main() {
	image_1829px = cv::imread("image1.jpg");
	image_3658px = cv::imread("image1_2x_res.jpg");
	cv::cvtColor(image_1829px, image_1829px_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(image_3658px, image_3658px_gray, cv::COLOR_BGR2GRAY);

	auto aruco_v5_detectFunc = [](const cv::Mat &input) {
		return aruco_v5_detect(input).first.count();
	};

	std::vector<TestEntry> testEntries = {
		{"aruco_v5", false, false, aruco_v5_detectFunc},
		{"aruco_v5", true,  false, aruco_v5_detectFunc},
		{"aruco_v5", false, true,  aruco_v5_detectFunc},
		{"aruco_v5", true,  true,  aruco_v5_detectFunc}
	};

	for(auto &test : testEntries) {
		runTestSuite(test.parallel, test.hires, test.func, test.name);
		fmt::println("Completed test {}\n", test.name);
	}

}