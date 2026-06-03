import cv2 as cv
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import controls


CANON_W = 900
CANON_H = 320

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

        quad = find_cassette_quad(frame)

        if quad is not None:
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