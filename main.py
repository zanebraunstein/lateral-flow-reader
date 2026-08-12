import csv
import os
import time

import cv2 as cv
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

import calibration as calib
import recorder
import strip
import viz


CSV_NAME = "signal_log.csv"

FONT = cv.FONT_HERSHEY_SIMPLEX
CALIBRATE_WINDOW = "Calibrate - click 4 corners of the results window"

# A lateral flow result is read at a fixed time point; stopping on a keypress
# makes runs incomparable and leaves the plateau undefined.
RUN_DURATION_S = 15 * 60

CSV_HEADER = [
    "time_seconds",
    "test_strength",
    "control_strength",
    "tc_ratio",
    "test_snr",
    "control_snr",
    "test_present",
    "control_present",
    "stable_test",
    "stable_control",
    # Density, in raw a* units
    "test_area",
    "control_area",
    "tc_area_ratio",
    "test_peak_a",
    "control_peak_a",
    "profile_scale"
]


def start_camera():
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={
            "size": (1920, 1080),
            "format": "RGB888"
        }
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(0.5)

    try:
        picam2.set_controls({
            "AfMode": controls.AfModeEnum.Continuous
        })
    except Exception as e:
        print("Autofocus unavailable:", e)

    return picam2


def csv_row(elapsed, result, stability):
    return [
        f"{elapsed:.2f}",
        f"{result.test.strength:.3f}",
        f"{result.control.strength:.3f}",
        f"{result.tc_ratio:.3f}",
        f"{result.test.snr:.3f}",
        f"{result.control.snr:.3f}",
        int(result.test.present),
        int(result.control.present),
        int(stability.stable_test),
        int(stability.stable_control),
        f"{result.test.area:.4f}",
        f"{result.control.area:.4f}",
        f"{result.tc_area_ratio:.4f}",
        f"{result.test.peak_a:.4f}",
        f"{result.control.peak_a:.4f}",
        f"{result.scale:.4f}"
    ]


def grab_frame(picam2):
    return cv.cvtColor(picam2.capture_array(), cv.COLOR_RGB2BGR)


def run_calibration_ui(picam2):
    """
    Interactive corner clicking on the live camera. Returns a Calibration, or
    None if cancelled. Band positions are learned from the cassette and
    labelled by the control side; press 'f' to flip it.
    """
    points = []
    control_side = "left"

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv.namedWindow(CALIBRATE_WINDOW)
    cv.setMouseCallback(CALIBRATE_WINDOW, on_mouse)

    try:
        while True:
            frame = grab_frame(picam2)
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
                f"f=flip  s=save  r=reset  q=cancel",
                (20, 30), FONT, 0.65, viz.COLOR_TEXT, 2
            )

            control_frac = test_frac = None

            if len(points) == 4:
                quad = strip.order_points(np.array(points, dtype=np.float32))
                probe = strip.analyze(frame, window_quad=quad)

                if probe is not None:
                    control_frac, test_frac = calib.learn_bands(
                        probe.profile, probe.candidates, control_side
                    )

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

                    cv.putText(disp, note, (20, 60), FONT, 0.7,
                               viz.COLOR_CONTROL if ok else viz.COLOR_SEARCHING, 2)

            cv.imshow(CALIBRATE_WINDOW, disp)

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                return None
            elif key == ord("r"):
                points = []
            elif key == ord("f"):
                control_side = "right" if control_side == "left" else "left"
            elif key == ord("s") and len(points) == 4:
                if control_frac is None or test_frac is None:
                    print("Both bands must be found before saving "
                          "(use a cassette showing both lines).")
                    continue

                h, w = frame.shape[:2]
                calibration = calib.from_points(
                    points, (w, h),
                    control_frac=control_frac, test_frac=test_frac
                )
                print(f"Calibrated: control@{control_frac:.2f} test@{test_frac:.2f} "
                      f"(control on {control_side})")
                return calibration

    finally:
        cv.destroyWindow(CALIBRATE_WINDOW)
        for extra in ("Strip preview", "Profile preview"):
            try:
                cv.destroyWindow(extra)
            except cv.error:
                pass


def preview_loop(picam2, calibration):
    """
    Live preview before recording. The user starts the run with 'g' (so the
    clock starts when the sample is applied), calibrates with 'c', or quits
    with 'q'. Returns (calibration, start) where start is True to record.
    """
    while True:
        frame = grab_frame(picam2)

        window_quad = calibration.window_quad if calibration else None
        bands = calibration.bands if calibration else None

        result = strip.analyze(frame, window_quad, bands)
        windows = viz.render(frame, result)

        disp = windows["Lateral Flow Reader"]
        state = "calibrated" if calibration else "NO calibration (press c)"
        cv.putText(disp, f"PREVIEW [{state}]   g=start run  c=calibrate  q=quit",
                   (20, 210), FONT, 0.7, viz.COLOR_TEXT, 2)

        for name, image in windows.items():
            cv.imshow(name, image)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            return calibration, False
        elif key == ord("g"):
            return calibration, True
        elif key == ord("c"):
            new = run_calibration_ui(picam2)
            if new is not None:
                new.save()
                calibration = new
                print("Saved calibration to", calib.CALIBRATION_PATH)


def record_run(picam2, run_dir, window_quad, bands):
    """
    The timed recording run. Stops after RUN_DURATION_S or on 'q'.
    """
    start_time = time.time()

    csv_file = open(os.path.join(run_dir, CSV_NAME), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

    stability = strip.StabilityTracker()
    run = recorder.RunRecorder(run_dir)

    stopped = "interrupted"

    try:
        while True:
            elapsed = time.time() - start_time

            if elapsed >= RUN_DURATION_S:
                stopped = f"completed ({RUN_DURATION_S / 60:.0f} min)"
                break

            frame = grab_frame(picam2)

            result = strip.analyze(frame, window_quad, bands)

            if result is not None:
                stability.update(result)

                writer.writerow(csv_row(elapsed, result, stability))

                # A run lasts minutes; never lose it to an unclean exit
                csv_file.flush()

                run.add(elapsed, result)

            windows = viz.render(
                frame,
                result,
                stability.stable_test,
                stability.stable_control
            )

            for name, image in windows.items():
                cv.imshow(name, image)

            if cv.waitKey(1) & 0xFF == ord("q"):
                stopped = f"stopped early at {elapsed / 60:.1f} min"
                break

    finally:
        # A failed save must not stop the camera from being released
        try:
            run.save()
        except Exception as e:
            print("Failed to write profiles.npz:", e)

        csv_file.close()

        print(f"Run {stopped}: {run.summary()}")


def main():
    picam2 = start_camera()

    try:
        calibration = calib.load()

        if calibration is not None:
            print("Loaded calibration from", calib.CALIBRATION_PATH)
        else:
            print("No calibration yet -- press 'c' in the preview to calibrate, "
                  "or 'g' to run with full-cassette detection.")

        calibration, start = preview_loop(picam2, calibration)

        if not start:
            print("Quit before starting a run.")
            return

        window_quad = calibration.window_quad if calibration else None
        bands = calibration.bands if calibration else None

        run_dir = recorder.create_run_dir()
        print("Recording run to", run_dir)

        record_run(picam2, run_dir, window_quad, bands)

    finally:
        cv.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
