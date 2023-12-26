#include "aruco_nano.h"
#include <chrono>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <fmt/core.h>

using namespace aruconano;
using namespace std::chrono;
using namespace std::chrono_literals;

using double_ms_dur = std::chrono::duration<double, std::milli>;

std::pair<double_ms_dur, std::vector<Marker>> detect(cv::Mat &inputMat) {
	std::vector<Marker> detectedMarkers;

	steady_clock::time_point begin = steady_clock::now();
	auto markers = MarkerDetector::detect(inputMat, 10, MarkerDetector::Dict::APRILTAG_36h11);
	double_ms_dur detectElapsed = steady_clock::now() - begin;

	return {detectElapsed, detectedMarkers};
}

constexpr int kWarmupLoops = 10;


int main() {
	auto image = cv::imread("image1.jpg");
	
	double mean = 0.0;
  const int iterations = 1000;

  for (int i = 0; i < iterations; i++) {
    auto start = high_resolution_clock::now();

    auto markers = aruconano::MarkerDetector::detect(
        image, 10, aruconano::MarkerDetector::APRILTAG_36h11);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    double milliseconds = duration.count() / 1000.0;
    mean += milliseconds;
    // std::cout << milliseconds << std::endl;
  }

  std::cout << "Mean: " << mean / iterations << "ms" << std::endl;

	// // warmup loops
	// fmt::println("Warming up...");
	// steady_clock::time_point warmupBegin = steady_clock::now();
	// std::vector<Marker> warmupMarkers;

	// for(int i = 0; i < kWarmupLoops; i++) {
	// 	auto ret = detect(image);
	// 	warmupMarkers = ret.second;
	// }
	// double_ms_dur warmupElapsed = steady_clock::now() - warmupBegin;
	// double warmupAvg = warmupElapsed.count() / kWarmupLoops;

	// fmt::println("Warmed up for 10 runs, {:.2f}ms total, {:.4f}ms avg", warmupElapsed.count(), warmupAvg);

	// cv::imwrite("output.png", image);
}