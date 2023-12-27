package org.photonvision;

import edu.wpi.first.cscore.CameraServerCvJNI;
import edu.wpi.first.util.CombinedRuntimeLoader;
import java.io.IOException;
import org.opencv.core.Core;

public class DetectorTest {

    public static boolean loadLibraries() {
        CameraServerCvJNI.Helper.setExtractOnStaticLoad(false);

        try {
            CombinedRuntimeLoader.loadLibraries(DetectorTest.class, Core.NATIVE_LIBRARY_NAME);

            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    // @Test
    // public void testThing() {
    //     DetectorTest.loadLibraries();
    //     System.load(
    //
    // "/home/matt/Documents/GitHub/fiducial-playground/build/libs/photonmiscjniJNI/shared/linuxx86-64/release/libphotonmiscjnijni.so");

    //     var mat = Imgcodecs.imread("image1.jpg");
    //     var out = new Mat();
    //     mat.copyTo(out);
    //     Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);

    //     for (int i = 0; i < 1; i++) {
    //         var start = System.nanoTime();
    //         var ret = List.of(ArucoNanoV5Detector.detect(mat.getNativeObjAddr(), 0));
    //         var end = System.nanoTime();
    //         var dt = (end - start) / 1e6;
    //         System.out.println(ret);
    //         System.out.println("Dt ms: " + dt);

    //         ArrayList<MatOfPoint> pts = new ArrayList<>();
    //         for (var tgt : ret) {
    //             var m = new MatOfPoint();
    //             // TODO
    //             pts.add(m);
    //         }
    //         Imgproc.polylines(out, pts, true, new Scalar(0, 255, 0), 2);

    //         for (var tgt : ret) {
    //             Imgproc.putText(
    //                     out,
    //                     "id " + tgt.id,
    //                     new Point(tgt.xCorners[0], tgt.yCorners[0]),
    //                     0,
    //                     1,
    //                     new Scalar(255, 255, 0),
    //                     2);
    //         }

    //         // HighGui.imshow("foo", out);
    //         // HighGui.waitKey(20000);
    //     }
    // }
}
