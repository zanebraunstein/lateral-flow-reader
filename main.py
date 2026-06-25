import cv2 as cv
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import controls


# Canonical cassette geometry
CANON_W = 900
CANON_H = 320

# Results window ROI in canonical warped cassette view
WINDOW_X0 = 0.55
WINDOW_Y0 = 0.22
WINDOW_X1 = 0.93
WINDOW_Y1 = 0.78

STRIP_Y0_FRAC = 0.44
STRIP_Y1_FRAC = 0.66

STRIP_X0_FRAC = 0.30
STRIP_X1_FRAC = 0.82


def order_points(pts):
    pts = pts.astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_cassette_quad(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    edges = cv.Canny(gray, 50, 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv.findContours(
        edges,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv.contourArea, reverse=True)

    h, w = frame.shape[:2]
    min_area = 0.03 * (h * w)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < min_area:
            continue

        peri = cv.arcLength(cnt, True)

        approx = cv.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        if len(approx) == 4 and cv.isContourConvex(approx):
            return order_points(
                approx.reshape(4, 2)
            )

    return None


def warp_cassette(frame, quad):
    dst = np.array([
        [0, 0],
        [CANON_W - 1, 0],
        [CANON_W - 1, CANON_H - 1],
        [0, CANON_H - 1]
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(
        quad.astype(np.float32),
        dst
    )

    return cv.warpPerspective(
        frame,
        M,
        (CANON_W, CANON_H)
    )

def extract_strip_roi(results_window):
    h, w = results_window.shape[:2]

    y0 = int(STRIP_Y0_FRAC * h)
    y1 = int(STRIP_Y1_FRAC * h)

    x0 = int(STRIP_X0_FRAC * w)
    x1 = int(STRIP_X1_FRAC * w)

    return results_window[y0:y1, x0:x1]

def redness_profile(strip_bgr):
    lab = cv.cvtColor(strip_bgr, cv.COLOR_BGR2LAB)

    a = lab[:, :, 1].astype(np.float32)

    # Collapse vertically into 1D profile
    profile = np.median(a, axis=0)

    # Smooth profile
    kernel = np.ones(21, dtype=np.float32) / 21.0
    profile = np.convolve(profile, kernel, mode="same")

    return profile

def draw_profile(profile, width=600, height=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    p = profile.astype(np.float32)

    p -= np.min(p)

    if np.max(p) > 1e-6:
        p /= np.max(p)

    xs = np.linspace(
        0,
        width - 1,
        len(p)
    ).astype(np.int32)

    ys = (
        height - 1
        - p * (height - 1)
    ).astype(np.int32)

    pts = np.stack([xs, ys], axis=1)

    cv.polylines(
        canvas,
        [pts],
        False,
        (255, 255, 255),
        2
    )

    return canvas


def main():
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

    while True:
        frame_rgb = picam2.capture_array()

        frame = cv.cvtColor(
            frame_rgb,
            cv.COLOR_RGB2BGR
        )

        disp = frame.copy()
        warped = None

        quad = find_cassette_quad(frame)

        if quad is not None:
            warped = warp_cassette(frame, quad)

            cv.polylines(
                disp,
                [quad.astype(np.int32)],
                True,
                (0, 255, 0),
                3
            )

            cv.putText(
                disp,
                "Cassette detected",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Results window ROI
            x0 = int(WINDOW_X0 * CANON_W)
            y0 = int(WINDOW_Y0 * CANON_H)
            x1 = int(WINDOW_X1 * CANON_W)
            y1 = int(WINDOW_Y1 * CANON_H)

            cv.rectangle(
                warped,
                (x0, y0),
                (x1, y1),
                (0, 255, 0),
                2
            )

            results_window = warped[y0:y1, x0:x1]

            rh, rw = results_window.shape[:2]

            sx0 = int(STRIP_X0_FRAC * rw)
            sx1 = int(STRIP_X1_FRAC * rw)

            sy0 = int(STRIP_Y0_FRAC * rh)
            sy1 = int(STRIP_Y1_FRAC * rh)

            cv.rectangle(
                results_window,
                (sx0, sy0),
                (sx1, sy1),
                (255, 0, 255),
                2
            )

            strip_roi = extract_strip_roi(results_window)

            profile = redness_profile(strip_roi)

            profile_vis = draw_profile(profile)

            cv.imshow(
                "Results Window",
                results_window
            )

            cv.imshow(
                "Strip ROI",
                strip_roi
            )

            cv.imshow(
                "Signal Profile",
                profile_vis
            )

        else:
            cv.putText(
                disp,
                "Searching...",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        if warped is not None:
            cv.imshow(
                "Warped Cassette",
                warped
            )

        cv.imshow(
            "Lateral Flow Reader",
            disp
        )

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()
    picam2.stop()


if __name__ == "__main__":
    main()