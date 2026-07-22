import csv
import os
import time

import cv2 as cv
from picamera2 import Picamera2
from libcamera import controls

import recorder
import strip
import viz


CSV_NAME = "signal_log.csv"

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


def main():
    run_dir = recorder.create_run_dir()
    print("Recording run to", run_dir)

    picam2 = start_camera()

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

            frame = cv.cvtColor(
                picam2.capture_array(),
                cv.COLOR_RGB2BGR
            )

            result = strip.analyze(frame)

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
        cv.destroyAllWindows()
        picam2.stop()

        print(f"Run {stopped}: {run.summary()}")


if __name__ == "__main__":
    main()
