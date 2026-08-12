"""
One-time calibration for a fixed camera rig.

Click the four corners of the results window (the bright membrane rectangle
containing the C and T lines) on the live camera view. The tool finds the two
band positions from the cassette itself and labels them from the control side
you set, so either cassette orientation works with no hand-tuned constants.

A live preview shows the warped strip and its signal profile with the bands
marked, so you can confirm before saving.

    python3 calibrate.py

Keys:  click 4 corners   f = flip control side   s = save   r = reset   q = quit

Calibrate with a cassette showing BOTH lines (a used positive works well) so
both band positions can be learned.

Writes calibration.json (and calibration.jpg). main.py picks it up automatically.
"""

import cv2 as cv
import numpy as np

import calibration as calib
import strip
import viz
from main import start_camera


FONT = cv.FONT_HERSHEY_SIMPLEX
WINDOW = "Calibrate - click 4 corners of the results window"


def learn_bands(result, control_side):
    """
    From an analysed frame, locate the two bands and assign control/test by the
    given side. Returns (control_frac, test_frac); either may be None.
    """
    n = len(result.profile)
    found = strip.dominant_two_bands(result.profile, result.candidates)

    if len(found) == 2:
        left, right = found
        control_idx, test_idx = (left, right) if control_side == "left" else (right, left)
    elif len(found) == 1:
        control_idx, test_idx = found[0], None
    else:
        control_idx, test_idx = None, None

    control_frac = None if control_idx is None else control_idx / n
    test_frac = None if test_idx is None else test_idx / n

    return control_frac, test_frac


def main():
    picam2 = start_camera()

    points = []
    control_side = "left"

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
                f"Corners {len(points)}/4   control={control_side}   "
                f"f=flip  s=save  r=reset  q=quit",
                (20, 30), FONT, 0.65, viz.COLOR_TEXT, 2
            )

            control_frac = test_frac = None

            # Once four corners are set, learn the bands and preview live.
            if len(points) == 4:
                quad = strip.order_points(np.array(points, dtype=np.float32))
                probe = strip.analyze(frame, window_quad=quad)

                if probe is not None:
                    control_frac, test_frac = learn_bands(probe, control_side)

                    result = strip.analyze(
                        frame, window_quad=quad, bands=(test_frac, control_frac)
                    )
                    previews = viz.render(frame, result)
                    cv.imshow("Strip preview", previews["Strip ROI"])
                    cv.imshow("Profile preview", previews["Signal Profile"])

                    ok = (
                        control_frac is not None
                        and result.control.snr >= strip.CONTROL_SNR_THRESHOLD
                    )
                    if control_frac is None:
                        note = "no bands found - check corners/lighting"
                    elif test_frac is None:
                        note = "only one line - calibrate with both lines showing"
                    else:
                        note = f"control SNR {result.control.snr:.1f} {'OK' if ok else 'LOW'}"

                    cv.putText(
                        disp, note, (20, 60), FONT, 0.7,
                        viz.COLOR_CONTROL if ok else viz.COLOR_SEARCHING, 2
                    )

            cv.imshow(WINDOW, disp)

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                points = []
            elif key == ord("f"):
                control_side = "right" if control_side == "left" else "left"
            elif key == ord("s") and len(points) == 4:
                if control_frac is None or test_frac is None:
                    print("Not saving: both bands must be found first "
                          "(use a cassette showing both lines).")
                    continue

                h, w = frame.shape[:2]
                calib.from_points(
                    points, (w, h),
                    control_frac=control_frac, test_frac=test_frac
                ).save()
                cv.imwrite("calibration.jpg", disp)
                saved = True
                print(f"Saved {calib.CALIBRATION_PATH}: control@{control_frac:.2f} "
                      f"test@{test_frac:.2f} (control on {control_side})")
                break

    finally:
        cv.destroyAllWindows()
        picam2.stop()

        if not saved:
            print("No calibration saved.")


if __name__ == "__main__":
    main()
