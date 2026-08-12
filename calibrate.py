"""
One-time calibration for a fixed camera rig.

Click the four corners of the results window (the bright membrane rectangle
containing the C and T lines) on the live camera view. A live preview shows
the warped strip and its signal profile, so you can confirm the bands are
found before saving.

    python3 calibrate.py

Keys:  click 4 corners   s = save   r = reset   q = quit

Writes calibration.json (and calibration.jpg, a snapshot of the marked frame).
Then run main.py, which picks the calibration up automatically.
"""

import cv2 as cv
import numpy as np

import calibration as calib
import strip
import viz
from main import start_camera


FONT = cv.FONT_HERSHEY_SIMPLEX
WINDOW = "Calibrate - click 4 corners of the results window"


def main():
    picam2 = start_camera()

    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv.namedWindow(WINDOW)
    cv.setMouseCallback(WINDOW, on_mouse)

    saved = False

    try:
        while True:
            frame = cv.cvtColor(picam2.capture_array(), cv.COLOR_RGB2BGR)
            disp = frame.copy()

            for i, point in enumerate(points):
                cv.circle(disp, point, 6, viz.COLOR_CONTROL, -1)
                cv.putText(disp, str(i + 1), (point[0] + 8, point[1] - 8),
                           FONT, 0.7, viz.COLOR_CONTROL, 2)

            if len(points) >= 2:
                cv.polylines(disp, [np.array(points, np.int32)],
                             len(points) == 4, viz.COLOR_CANDIDATE, 2)

            cv.putText(
                disp,
                f"Corners {len(points)}/4    s=save  r=reset  q=quit",
                (20, 30), FONT, 0.7, viz.COLOR_TEXT, 2
            )

            # Once four corners are set, preview the strip and profile live so
            # the user can confirm the bands land before committing.
            if len(points) == 4:
                quad = strip.order_points(np.array(points, dtype=np.float32))
                result = strip.analyze(frame, window_quad=quad)

                if result is not None:
                    previews = viz.render(frame, result)
                    cv.imshow("Strip preview", previews["Strip ROI"])
                    cv.imshow("Profile preview", previews["Signal Profile"])

                    ok = result.control.snr >= strip.CONTROL_SNR_THRESHOLD
                    cv.putText(
                        disp,
                        f"Control SNR {result.control.snr:.1f} "
                        f"{'OK' if ok else 'LOW - adjust corners/lighting'}",
                        (20, 60), FONT, 0.7,
                        viz.COLOR_CONTROL if ok else viz.COLOR_SEARCHING, 2
                    )

            cv.imshow(WINDOW, disp)

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                points = []
            elif key == ord("s") and len(points) == 4:
                h, w = frame.shape[:2]
                calib.from_points(points, (w, h)).save()
                cv.imwrite("calibration.jpg", disp)
                saved = True
                print(f"Saved {calib.CALIBRATION_PATH} and calibration.jpg")
                break

    finally:
        cv.destroyAllWindows()
        picam2.stop()

        if not saved:
            print("No calibration saved.")


if __name__ == "__main__":
    main()
