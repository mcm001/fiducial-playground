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

package org.photonvision;

import java.util.Arrays;
import org.opencv.core.Point;

public final class ArucoNanoV5Detector {
    public static class DetectionResult {
        public DetectionResult(double[] corns, int id) {
            this.corners =
                    new Point[] {
                        new Point(corns[0], corns[1]),
                        new Point(corns[2], corns[3]),
                        new Point(corns[4], corns[5]),
                        new Point(corns[6], corns[7])
                    };
            this.id = id;
        }

        final Point[] corners;
        final int id;

        @Override
        public String toString() {
            return "DetectionResult [corners=" + Arrays.toString(corners) + ", id=" + id + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + Arrays.hashCode(corners);
            result = prime * result + id;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null) return false;
            if (getClass() != obj.getClass()) return false;
            DetectionResult other = (DetectionResult) obj;
            if (!Arrays.equals(corners, other.corners)) return false;
            if (id != other.id) return false;
            return true;
        }
    }

    public static native DetectionResult[] detect(long matPtr, int dictionary);
}
