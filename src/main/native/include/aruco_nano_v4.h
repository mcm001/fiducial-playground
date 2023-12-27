/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/** This is the ArucoNano library. A header only library that includes what 99%
 * of users need in a single header file. We compress all our knowledge in less
 * than 200lines of code that you can simply drop into your project.
 *
 * The library detects markers of dictionary ARUCO_MIP_36h12
 * (https://sourceforge.net/projects/aruco/files/aruco_mip_36h12_dict.zip/download)
 * Simply add this file to your project to start enjoying Aruco.
 *
 *
 * Example of Usage:
 *
#include "aruco_nano.h"

 * int main(){
 *   auto image=cv::imread("/path/to/image");
 *   auto markers=aruconano::MarkerDetector::detect(image);
 *   for(const auto &m:markers)
 *      m.draw(image);
 *    cv::imwrite("/path/to/out.png",image);
 *
 *   //now, compute R and T vectors
 *   cv::Mat camMatrix,distCoeff;
 *   float markerSize=0.05;//5cm
 *   //read CamMatrix and DistCoeffs from calibration std::FILE ....
 *    for(const auto &m:markers)
 *      auto r_t=m.estimatePose(camMatrix,distCoeff,markerSize);
 * }
 *
 * If you use this file in your research, you must cite:
 *
 * 1."Speeded up detection of squared fiducial markers", Francisco
 * J.Romero-Ramirez, Rafael Muñoz-Salinas, Rafael Medina-Carnicer, Image and
 * Vision Computing, vol 76, pages 38-47, year 2018 2."Generation of fiducial
 * marker dictionaries using mixed integer linear programming",S.
 * Garrido-Jurado, R. Muñoz Salinas, F.J. Madrid-Cuevas, R. Medina-Carnicer,
 * Pattern Recognition:51, 481-491,2016
 *
 *
 * You can freely use the code in your commercial products.
 *
 * Why is the code obfuscated? Because we do not want copycats taking credit for
 * our work.
 *
 */
/*
Copyright 2022 Rafael Muñoz Salinas. All rights reserved.
  This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef FIDUCIAL_PLAYGROUND_SRC_MAIN_NATIVE_INCLUDE_ARUCO_NANO_V4_H_
#define FIDUCIAL_PLAYGROUND_SRC_MAIN_NATIVE_INCLUDE_ARUCO_NANO_V4_H_
#define ArucoNanoVersion 4
#include <bitset>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
namespace aruconano {
class Marker : public std::vector<cv::Point2f> {
public:
  int id = -1;
  inline void draw(cv::Mat &image,
                   const cv::Scalar color = cv::Scalar(0, 0, 255)) const;
  inline std::pair<cv::Mat, cv::Mat>
  estimatePose(cv::Mat cameraMatrix, cv::Mat distCoeffs,
               double markerSize = 1.0f) const;
};
class MarkerDetector {
public:
  static inline std::vector<Marker>
  detect(const cv::Mat &img, unsigned int maxAttemptsPerCandidate = 10) {
    return internalDetect(img, maxAttemptsPerCandidate);
  }

private:
  static inline std::vector<Marker>
  internalDetect(const cv::Mat &inputMat,
                 unsigned int maxAttemptsPerCandidate = 10);
  static inline Marker looksLikeProcessMarker(const Marker &marker);
  static inline float linearSampleImage(const cv::Mat &mat,
                                        const cv::Point2f &point);
  static inline int findMarkerId(const cv::Mat &mat, int &numRotationsNeeded,
                                 const std::vector<uint64_t> &dictionary);
  static inline int sideLengthOfMarker(const std::vector<cv::Point2f> &points);
};
namespace _private {
struct PerspectiveTransformer {
  PerspectiveTransformer(const std::vector<cv::Point2f> &dstPoints) {
    std::vector<cv::Point2f> srcPoints = {cv::Point2f(0, 0), cv::Point2f(1, 0),
                                          cv::Point2f(1, 1), cv::Point2f(0, 1)};
    transformMat = cv::getPerspectiveTransform(srcPoints, dstPoints);
  }
  cv::Point2f operator()(const cv::Point2f &point) {
    double *matPointer = transformMat.ptr<double>(0);
    double xPrime =
        matPointer[0] * point.x + matPointer[1] * point.y + matPointer[2];
    double yPrime =
        matPointer[3] * point.x + matPointer[4] * point.y + matPointer[5];
    double w =
        matPointer[6] * point.x + matPointer[7] * point.y + matPointer[8];
    return cv::Point2f(xPrime / w, yPrime / w);
  }
  cv::Mat transformMat;
};
} // namespace _private
std::vector<Marker>
MarkerDetector::internalDetect(const cv::Mat &inputMat,
                               unsigned int maxAttemptsPerCandidate) {
  if (maxAttemptsPerCandidate == 0)
    maxAttemptsPerCandidate = 1;
  cv::Mat grayMat, thresholdedMat;
  std::vector<Marker> detectedMarkers;
  std::vector<uint64_t> aruco_36h12_codes = {
      0xd2b63a09dUL, 0x6001134e5UL, 0x1206fbe72UL, 0xff8ad6cb4UL,
      0x85da9bc49UL, 0xb461afe9cUL, 0x6db51fe13UL, 0x5248c541fUL,
      0x8f34503UL,   0x8ea462eceUL, 0xeac2be76dUL, 0x1af615c44UL,
      0xb48a49f27UL, 0x2e4e1283bUL, 0x78b1f2fa8UL, 0x27d34f57eUL,
      0x89222fff1UL, 0x4c1669406UL, 0xbf49b3511UL, 0xdc191cd5dUL,
      0x11d7c3f85UL, 0x16a130e35UL, 0xe29f27effUL, 0x428d8ae0cUL,
      0x90d548477UL, 0x2319cbc93UL, 0xc3b0c3dfcUL, 0x424bccc9UL,
      0x2a081d630UL, 0x762743d96UL, 0xd0645bf19UL, 0xf38d7fd60UL,
      0xc6cbf9a10UL, 0x3c1be7c65UL, 0x276f75e63UL, 0x4490a3f63UL,
      0xda60acd52UL, 0x3cc68df59UL, 0xab46f9daeUL, 0x88d533d78UL,
      0xb6d62ec21UL, 0xb3c02b646UL, 0x22e56d408UL, 0xac5f5770aUL,
      0xaaa993f66UL, 0x4caa07c8dUL, 0x5c9b4f7b0UL, 0xaa9ef0e05UL,
      0x705c5750UL,  0xac81f545eUL, 0x735b91e74UL, 0x8cc35cee4UL,
      0xe44694d04UL, 0xb5e121de0UL, 0x261017d0fUL, 0xf1d439eb5UL,
      0xa1a33ac96UL, 0x174c62c02UL, 0x1ee27f716UL, 0x8b1c5ece9UL,
      0x6a05b0c6aUL, 0xd0568dfcUL,  0x192d25e5fUL, 0x1adbeccc8UL,
      0xcfec87f00UL, 0xd0b9dde7aUL, 0x88dcef81eUL, 0x445681cb9UL,
      0xdbb2ffc83UL, 0xa48d96df1UL, 0xb72cc2e7dUL, 0xc295b53fUL,
      0xf49832704UL, 0x9968edc29UL, 0x9e4e1af85UL, 0x8683e2d1bUL,
      0x810b45c04UL, 0x6ac44bfe2UL, 0x645346615UL, 0x3990bd598UL,
      0x1c9ed0f6aUL, 0xc26729d65UL, 0x83993f795UL, 0x3ac05ac5dUL,
      0x357adff3bUL, 0xd5c05565UL,  0x2f547ef44UL, 0x86c115041UL,
      0x640fd9e5fUL, 0xce08bbcf7UL, 0x109bb343eUL, 0xc21435c92UL,
      0x35b4dfce4UL, 0x459752cf2UL, 0xec915b82cUL, 0x51881eed0UL,
      0x2dda7dc97UL, 0x2e0142144UL, 0x42e890f99UL, 0x9a8856527UL,
      0x8e80d9d80UL, 0x891cbcf34UL, 0x25dd82410UL, 0x239551d34UL,
      0x8fe8f0c70UL, 0x94106a970UL, 0x82609b40cUL, 0xfc9caf36UL,
      0x688181d11UL, 0x718613c08UL, 0xf1ab7629UL,  0xa357bfc18UL,
      0x4c03b7a46UL, 0x204dedce6UL, 0xad6300d37UL, 0x84cc4cd09UL,
      0x42160e5c4UL, 0x87d2adfa8UL, 0x7850e7749UL, 0x4e750fc7cUL,
      0xbf2e5dfdaUL, 0xd88324da5UL, 0x234b52f80UL, 0x378204514UL,
      0xabdf2ad53UL, 0x365e78ef9UL, 0x49caa6ca2UL, 0x3c39ddf3UL,
      0xc68c5385dUL, 0x5bfcbbf67UL, 0x623241e21UL, 0xabc90d5ccUL,
      0x388c6fe85UL, 0xda0e2d62dUL, 0x10855dfe9UL, 0x4d46efd6bUL,
      0x76ea12d61UL, 0x9db377d3dUL, 0xeed0efa71UL, 0xe6ec3ae2fUL,
      0x441faee83UL, 0xba19c8ff5UL, 0x313035eabUL, 0x6ce8f7625UL,
      0x880dab58dUL, 0x8d3409e0dUL, 0x2be92ee21UL, 0xd60302c6cUL,
      0x469ffc724UL, 0x87eebeed3UL, 0x42587ef7aUL, 0x7a8cc4e52UL,
      0x76a437650UL, 0x999e41ef4UL, 0x7d0969e42UL, 0xc02baf46bUL,
      0x9259f3e47UL, 0x2116a1dc0UL, 0x9f2de4d84UL, 0xeffac29UL,
      0x7b371ff8cUL, 0x668339da9UL, 0xd010aee3fUL, 0x1cd00b4c0UL,
      0x95070fc3bUL, 0xf84c9a770UL, 0x38f863d76UL, 0x3646ff045UL,
      0xce1b96412UL, 0x7a5d45da8UL, 0x14e00ef6cUL, 0x5e95abfd8UL,
      0xb2e9cb729UL, 0x36c47dd7UL,  0xb8ee97c6bUL, 0xe9e8f657UL,
      0xd4ad2ef1aUL, 0x8811c7f32UL, 0x47bde7c31UL, 0x3adadfb64UL,
      0x6e5b28574UL, 0x33e67cd91UL, 0x2ab9fdd2dUL, 0x8afa67f2bUL,
      0xe6a28fc5eUL, 0x72049cdbdUL, 0xae65dac12UL, 0x1251a4526UL,
      0x1089ab841UL, 0xe2f096ee0UL, 0xb0caee573UL, 0xfd6677e86UL,
      0x444b3f518UL, 0xbe8b3a56aUL, 0x680a75cfcUL, 0xac02baea8UL,
      0x97d815e1cUL, 0x1d4386e08UL, 0x1a14f5b0eUL, 0xe658a8d81UL,
      0xa3868efa7UL, 0x3668a9673UL, 0xe8fc53d85UL, 0x2e2b7edd5UL,
      0x8b2470f13UL, 0xf69795f32UL, 0x4589ffc8eUL, 0x2e2080c9cUL,
      0x64265f7dUL,  0x3d714dd10UL, 0x1692c6ef1UL, 0x3e67f2f49UL,
      0x5041dad63UL, 0x1a1503415UL, 0x64c18c742UL, 0xa72eec35UL,
      0x1f0f9dc60UL, 0xa9559bc67UL, 0xf32911d0dUL, 0x21c0d4ffcUL,
      0xe01cef5b0UL, 0x4e23a3520UL, 0xaa4f04e49UL, 0xe1c4fcc43UL,
      0x208e8f6e8UL, 0x8486774a5UL, 0x9e98c7558UL, 0x2c59fb7dcUL,
      0x9446a4613UL, 0x8292dcc2eUL, 0x4d61631UL,   0xd05527809UL,
      0xa0163852dUL, 0x8f657f639UL, 0xcca6c3e37UL, 0xcb136bc7aUL,
      0xfc5a83e53UL, 0x9aa44fc30UL, 0xbdec1bd3cUL, 0xe020b9f7cUL,
      0x4b8f35fb0UL, 0xb8165f637UL, 0x33dc88d69UL, 0x10a2f7e4dUL,
      0xc8cb5ff53UL, 0xde259ff6bUL, 0x46d070dd4UL, 0x32d3b9741UL,
      0x7075f1c04UL, 0x4d58dbea0UL};
  if (inputMat.channels() == 3)
    cv::cvtColor(inputMat, grayMat, cv::COLOR_BGR2GRAY);
  else
    grayMat = inputMat;
  cv::adaptiveThreshold(grayMat, thresholdedMat, 255.,
                        cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 7,
                        7);
  std::vector<std::vector<cv::Point>> foundCountours;
  std::vector<cv::Point> approxContour;
  cv::RNG rng;
  cv::findContours(thresholdedMat, foundCountours, cv::noArray(), cv::RETR_LIST,
                   cv::CHAIN_APPROX_NONE);
  cv::Mat sampledMarker(8, 8, CV_8UC1);

  for (unsigned int contourIndex = 0; contourIndex < foundCountours.size();
       contourIndex++) {
    if (50 > int(foundCountours[contourIndex].size()))
      continue;
    cv::approxPolyDP(foundCountours[contourIndex], approxContour,
                     double(foundCountours[contourIndex].size()) * 0.05, true);
    if (approxContour.size() != 4 || !cv::isContourConvex(approxContour))
      continue;
    Marker marker;
    for (int pointIndex = 0; pointIndex < 4; pointIndex++)
      marker.push_back(cv::Point2f(approxContour[pointIndex].x,
                                   approxContour[pointIndex].y));
    marker = looksLikeProcessMarker(marker);
    for (int i = 0; i < maxAttemptsPerCandidate && marker.id == -1; i++) {
      auto markerCopy = marker;
      if (i != 0)
        for (int pointIndex = 0; pointIndex < 4; pointIndex++) {
          markerCopy[pointIndex].x += rng.gaussian(0.75);
          markerCopy[pointIndex].y += rng.gaussian(0.75);
        }

      int totalSampledValue = 0;
      _private::PerspectiveTransformer transformer(markerCopy);
      for (int row = 0; row < sampledMarker.rows; row++) {
        for (int col = 0; col < sampledMarker.cols; col++) {
          auto linearlySampled = uchar(
              0.5 +
              linearSampleImage(
                  grayMat, transformer(cv::Point2f(
                               float(col + 0.5) / float(sampledMarker.cols),
                               float(row + 0.5) / float(sampledMarker.rows)))));
          sampledMarker.at<uchar>(row, col) = linearlySampled;
          totalSampledValue += linearlySampled;
        }
      }
      double thresholdCutoff = double(totalSampledValue) /
                               double(sampledMarker.cols * sampledMarker.rows);

      cv::threshold(sampledMarker, sampledMarker, thresholdCutoff, 255,
                    cv::THRESH_BINARY);
      int numRotationsNeeded = 0;
      marker.id =
          findMarkerId(sampledMarker, numRotationsNeeded, aruco_36h12_codes);
      if (marker.id == -1)
        continue;
      std::rotate(marker.begin(), marker.begin() + 4 - numRotationsNeeded,
                  marker.end());
    }
    if (marker.id != -1)
      detectedMarkers.push_back(marker);
  }
  std::sort(detectedMarkers.begin(), detectedMarkers.end(),
            [](const Marker &first, const Marker &second) {
              if (first.id < second.id)
                return true;
              else if (first.id == second.id)
                return sideLengthOfMarker(first) > sideLengthOfMarker(second);
              else
                return false;
            });

  auto uniqueEnd = std::unique(detectedMarkers.begin(), detectedMarkers.end(),
                               [](const Marker &first, const Marker &second) {
                                 return first.id == second.id;
                               });
  detectedMarkers.resize(std::distance(detectedMarkers.begin(), uniqueEnd));

  if (detectedMarkers.size() > 0) {
    int winSize = 4 * float(grayMat.cols) / float(grayMat.cols) + 0.5;
    std::vector<cv::Point2f> pointsToSubPix;
    for (const auto &marker : detectedMarkers) {
      pointsToSubPix.insert(pointsToSubPix.end(), marker.begin(), marker.end());
    }

    cv::cornerSubPix(
        grayMat, pointsToSubPix, cv::Size(winSize, winSize), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12,
                         0.005));
    for (unsigned int markerIndex = 0; markerIndex < detectedMarkers.size();
         markerIndex++)
      for (int pointIndex = 0; pointIndex < 4; pointIndex++)
        detectedMarkers[markerIndex][pointIndex] =
            pointsToSubPix[markerIndex * 4 + pointIndex];
  }
  return detectedMarkers;
}
int MarkerDetector::sideLengthOfMarker(const std::vector<cv::Point2f> &points) {
  int ret = 0;
  for (size_t i = 0; i < points.size(); i++)
    // euclidean distance
    ret += cv::norm(points[i] - points[(i + 1) % points.size()]);
  return ret;
}
int MarkerDetector::findMarkerId(const cv::Mat &input, int &numRotationsNeeded,
                                 const std::vector<uint64_t> &dictionary) {
  auto decodeMarkerToId = [](const cv::Mat &mat) {
    std::bitset<64> bitset;
    int counter = 0;
    for (int i = mat.rows - 1; i >= 0; i--)
      for (int j = mat.cols - 1; j >= 0; j--)
        bitset[counter++] = mat.at<uchar>(i, j);
    return bitset.to_ullong();
  };

  auto rotate = [](const cv::Mat &mat) {
    cv::Mat ret(mat.size(), mat.type());
    for (int i = 0; i < mat.rows; i++)
      for (int j = 0; j < mat.cols; j++)
        ret.at<uchar>(i, j) = mat.at<uchar>(mat.cols - j - 1, i);
    return ret;
  };

  // checks for valid border
  for (int y = 0; y < input.cols; y++) {
    if (input.at<uchar>(0, y) != 0)
      return -1;
    if (input.at<uchar>(input.rows - 1, y) != 0)
      return -1;
    if (input.at<uchar>(y, 0) != 0)
      return -1;
    if (input.at<uchar>(y, input.cols - 1) != 0)
      return -1;
  }

  // discards the border
  cv::Mat matToConsider(input.cols - 2, input.rows - 2, CV_8UC1);
  for (int i = 0; i < matToConsider.rows; i++)
    for (int j = 0; j < matToConsider.cols; j++)
      matToConsider.at<uchar>(i, j) = input.at<uchar>(i + 1, j + 1);

  numRotationsNeeded = 0;
  do {
    auto decoded = decodeMarkerToId(matToConsider);
    for (size_t dictIndex = 0; dictIndex < dictionary.size(); dictIndex++)
      if (dictionary[dictIndex] == decoded)
        return dictIndex;
    matToConsider = rotate(matToConsider);
    numRotationsNeeded++;
  } while (numRotationsNeeded < 4);
  return -1;
}
float MarkerDetector::linearSampleImage(const cv::Mat &mat,
                                        const cv::Point2f &point) {
  float pointX = int(point.x);
  float pointY = int(point.y);
  if (pointX < 0 || pointX >= mat.cols - 1 || pointY < 0 ||
      pointY >= mat.rows - 1)
    return 0;
  const uchar *mainRow = mat.ptr<uchar>(pointY);
  const uchar *rowAfter = mat.ptr<uchar>(pointY + 1);

  float givenPoint = float(mainRow[int(pointX)]);
  float pointToRight = float(mainRow[int(pointX + 1)]);
  float pointBelow = float(rowAfter[int(pointX)]);
  float pointBelowAndRight = float(rowAfter[int(pointX + 1)]);

  float first = float(pointX + 1.f - point.x) * givenPoint +
                (point.x - pointX) * pointToRight;
  float second = float(pointX + 1.f - point.x) * pointBelow +
                 (point.x - pointX) * pointBelowAndRight;
  return (pointY + 1 - point.y) * first + (point.y - pointY) * second;
}
Marker MarkerDetector::looksLikeProcessMarker(const Marker &marker) {
  Marker temp = marker;
  double a = temp[1].x - temp[0].x;
  double b = temp[1].y - temp[0].y;
  double c = temp[2].x - temp[0].x;
  double d = temp[2].y - temp[0].y;
  double e = (a * d) - (b * c);
  if (e < 0.0)
    std::swap(temp[1], temp[3]);
  return temp;
}
std::pair<cv::Mat, cv::Mat> Marker::estimatePose(cv::Mat cameraMatrix,
                                                 cv::Mat distCoeffs,
                                                 double markerSize) const {
  std::vector<cv::Point3d> markerCorners = {
      {-markerSize / 2.f, markerSize / 2.f, 0.f},
      {markerSize / 2.f, markerSize / 2.f, 0.f},
      {markerSize / 2.f, -markerSize / 2.f, 0.f},
      {-markerSize / 2.f, -markerSize / 2.f, 0.f}};
  cv::Mat Rvec, Tvec;
  cv::solvePnP(markerCorners, *this, cameraMatrix, distCoeffs, Rvec, Tvec,
               false, cv::SOLVEPNP_IPPE);
  return {Rvec, Tvec};
}
void Marker::draw(cv::Mat &in, const cv::Scalar color) const {
  auto _to_string = [](int i) {
    std::stringstream str;
    str << i;
    return str.str();
  };
  float flineWidth = std::max(1.f, std::min(5.f, float(in.cols) / 500.f));
  int lineWidth = std::round(flineWidth);
  for (int i = 0; i < 4; i++)
    cv::line(in, (*this)[i], (*this)[(i + 1) % 4], color, lineWidth);
  auto p2 = cv::Point2f(2.f * static_cast<float>(lineWidth),
                        2.f * static_cast<float>(lineWidth));
  cv::rectangle(in, (*this)[0] - p2, (*this)[0] + p2,
                cv::Scalar(0, 0, 255, 255), -1);
  cv::rectangle(in, (*this)[1] - p2, (*this)[1] + p2,
                cv::Scalar(0, 255, 0, 255), lineWidth);
  cv::rectangle(in, (*this)[2] - p2, (*this)[2] + p2,
                cv::Scalar(255, 0, 0, 255), lineWidth);
  cv::Point2f cent(0, 0);
  for (auto &p : *this)
    cent += p;
  cent /= 4;
  float fsize = std::min(3.0f, flineWidth * 0.75f);
  cv::putText(in, _to_string(id), cent - cv::Point2f(10 * flineWidth, 0),
              cv::FONT_HERSHEY_SIMPLEX, fsize,
              cv::Scalar(255, 255, 255) - color, lineWidth, cv::LINE_AA);
}
} // namespace aruconano
#endif // FIDUCIAL_PLAYGROUND_SRC_MAIN_NATIVE_INCLUDE_ARUCO_NANO_V4_H_
