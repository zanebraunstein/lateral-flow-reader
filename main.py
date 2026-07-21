import csv
import time

import cv2 as cv
from picamera2 import Picamera2
from libcamera import controls

import strip
import viz


CSV_PATH = "signal_log.csv"

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
    "stable_control"
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
        int(stability.stable_control)
    ]


def main():
    picam2 = start_camera()

    start_time = time.time()

    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

    stability = strip.StabilityTracker()

    try:
        while True:
            frame = cv.cvtColor(
                picam2.capture_array(),
                cv.COLOR_RGB2BGR
            )

            result = strip.analyze(frame)

            if result is not None:
                stability.update(result)

                elapsed = time.time() - start_time

                writer.writerow(csv_row(elapsed, result, stability))

                # A run lasts minutes; never lose it to an unclean exit
                csv_file.flush()

            windows = viz.render(
                frame,
                result,
                stability.stable_test,
                stability.stable_control
            )

            for name, image in windows.items():
                cv.imshow(name, image)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        csv_file.close()
        cv.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
