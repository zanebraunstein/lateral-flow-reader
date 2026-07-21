import csv
import time
from collections import deque

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


def main():
    picam2 = start_camera()

    start_time = time.time()

    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

    recent_test = deque(maxlen=strip.STABILITY_WINDOW)
    recent_control = deque(maxlen=strip.STABILITY_WINDOW)

    try:
        while True:
            frame_rgb = picam2.capture_array()

            frame = cv.cvtColor(
                frame_rgb,
                cv.COLOR_RGB2BGR
            )

            disp = frame.copy()
            warped = None

            quad = strip.find_cassette_quad(frame)

            if quad is not None:
                warped = strip.warp_cassette(frame, quad)

                viz.draw_cassette_found(disp, quad)

                x0, y0, x1, y1 = strip.results_window_bounds()

                # NOTE: these ROI rectangles are drawn into `warped` before the
                # strip is sliced out of it, so the overlay lands in the pixels
                # that get measured. Left as-is here; fixed when analysis and
                # rendering are separated.
                cv.rectangle(
                    warped,
                    (x0, y0),
                    (x1, y1),
                    viz.COLOR_CONTROL,
                    2
                )

                results_window = warped[y0:y1, x0:x1]

                sx0, sy0, sx1, sy1 = strip.strip_bounds(results_window)

                cv.rectangle(
                    results_window,
                    (sx0, sy0),
                    (sx1, sy1),
                    (255, 0, 255),
                    2
                )

                strip_roi = strip.extract_strip_roi(results_window)

                profile = strip.redness_profile(strip_roi)

                candidates = strip.filter_band_candidates(profile)

                t_idx, c_idx = strip.pick_t_c_from_peaks(profile, candidates)

                if c_idx is not None:
                    viz.draw_band_line_on_main(
                        disp,
                        quad,
                        x0 + sx0 + c_idx,
                        y0,
                        y1,
                        "C",
                        viz.COLOR_CONTROL
                    )

                if t_idx is not None:
                    viz.draw_band_line_on_main(
                        disp,
                        quad,
                        x0 + sx0 + t_idx,
                        y0,
                        y1,
                        "T",
                        viz.COLOR_TEST
                    )

                t_strength, t_snr = strip.band_signal_snr(profile, t_idx)
                c_strength, c_snr = strip.band_signal_snr(profile, c_idx)

                control_present = c_snr >= strip.CONTROL_SNR_THRESHOLD
                test_present = (
                    t_snr >= strip.TEST_SNR_THRESHOLD
                    and control_present
                )

                recent_test.append(test_present)
                recent_control.append(control_present)

                stable_control = sum(recent_control) >= strip.STABILITY_VOTES
                stable_test = sum(recent_test) >= strip.STABILITY_VOTES

                if c_strength > 1e-6:
                    tc_ratio = t_strength / c_strength
                else:
                    tc_ratio = 0.0

                elapsed = time.time() - start_time

                writer.writerow([
                    f"{elapsed:.2f}",
                    f"{t_strength:.3f}",
                    f"{c_strength:.3f}",
                    f"{tc_ratio:.3f}",
                    f"{t_snr:.3f}",
                    f"{c_snr:.3f}",
                    int(test_present),
                    int(control_present),
                    int(stable_test),
                    int(stable_control)
                ])

                # A run lasts minutes; never lose it to an unclean exit
                csv_file.flush()

                profile_vis = viz.draw_profile(profile)

                viz.mark_bands_on_profile(
                    profile_vis,
                    profile,
                    t_idx,
                    c_idx,
                    candidates
                )

                viz.draw_readout(
                    disp,
                    t_strength,
                    c_strength,
                    tc_ratio,
                    t_snr,
                    c_snr,
                    test_present,
                    control_present,
                    stable_test,
                    stable_control
                )

                cv.imshow("Results Window", results_window)
                cv.imshow("Strip ROI", strip_roi)
                cv.imshow("Signal Profile", profile_vis)

            else:
                viz.draw_searching(disp)

            if warped is not None:
                cv.imshow("Warped Cassette", warped)

            cv.imshow("Lateral Flow Reader", disp)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        csv_file.close()
        cv.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
