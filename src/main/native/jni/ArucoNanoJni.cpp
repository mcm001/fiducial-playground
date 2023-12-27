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

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include "aruco_nano.h"
#include "org_photonvision_ArucoNanoV5Detector.h"

using namespace aruconano;

#define WPI_JNI_MAKEJARRAY(T, F)                                               \
  inline T##Array MakeJ##F##Array(JNIEnv *env, T *data, size_t size) {         \
    T##Array jarr = env->New##F##Array(size);                                  \
    if (!jarr) {                                                               \
      return nullptr;                                                          \
    }                                                                          \
    env->Set##F##ArrayRegion(jarr, 0, size, data);                             \
    return jarr;                                                               \
  }

WPI_JNI_MAKEJARRAY(jboolean, Boolean)
WPI_JNI_MAKEJARRAY(jbyte, Byte)
WPI_JNI_MAKEJARRAY(jshort, Short)
WPI_JNI_MAKEJARRAY(jlong, Long)
WPI_JNI_MAKEJARRAY(jfloat, Float)
WPI_JNI_MAKEJARRAY(jdouble, Double)

#undef WPI_JNI_MAKEJARRAY

/**
 * Finds a class and keeps it as a global reference.
 *
 * Use with caution, as the destructor does NOT call DeleteGlobalRef due to
 * potential shutdown issues with doing so.
 */
class JClass {
public:
  JClass() = default;

  JClass(JNIEnv *env, const char *name) {
    jclass local = env->FindClass(name);
    if (!local) {
      return;
    }
    m_cls = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
  }

  void free(JNIEnv *env) {
    if (m_cls) {
      env->DeleteGlobalRef(m_cls);
    }
    m_cls = nullptr;
  }

  explicit operator bool() const { return m_cls; }

  operator jclass() const { return m_cls; }

protected:
  jclass m_cls = nullptr;
};

JClass detectionResultClass;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv((void **)(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  detectionResultClass =
      JClass(env, "org/photonvision/ArucoNanoV5Detector$DetectionResult");

  if (!detectionResultClass) {
    std::printf("Couldn't find class!");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

static jobject MakeJObject(JNIEnv *env, const Marker &detect) {
  // TODO refactor into static refs
  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "([DI)V");

  jdouble corners[8]; // = new jdouble[8]{};
  for (int i = 0; i < 4; i++) {
    corners[i * 2] = detect[i].x;
    corners[i * 2 + 1] = detect[i].y;
  }
  jdoubleArray carr = MakeJDoubleArray(env, corners, 8);

  // Actually call the constructor
  return env->NewObject(detectionResultClass, constructor, carr, detect.id);
}

/*
 * Class:     org_photonvision_ArucoNanoV5Detector
 * Method:    detect
 * Signature: (JI)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_ArucoNanoV5Detector_detect
  (JNIEnv *env, jclass, jlong matPtr, jint dict)
{
  cv::Mat *mat = reinterpret_cast<cv::Mat *>(matPtr);
  unsigned int maxAttemptsPerCandidate = 10;
  auto markers = MarkerDetector::detect(*mat, maxAttemptsPerCandidate,
                                        TagDicts::APRILTAG_36h11);

  // Todo extract
  jobjectArray jarr =
      env->NewObjectArray(markers.size(), detectionResultClass, nullptr);
  for (size_t i = 0; i < markers.size(); i++) {
    jobject obj = MakeJObject(env, markers[i]);
    env->SetObjectArrayElement(jarr, i, obj);
  }

  return jarr;
}
