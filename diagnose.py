import time
import numpy as np
import cv2 as cv
import calibration as calib, strip, main

picam2 = main.start_camera()
time.sleep(1.5)
cal = calib.load()

N, cdet, tdet = 30, 0, 0
csn, tsn, sc, last = [], [], [], None
for _ in range(N):
    last = main.grab_frame(picam2)
    r = strip.analyze(last, cal.window_quad, cal.bands)
    cdet += r.control.idx is not None
    tdet += r.test.idx is not None
    csn.append(r.control.snr); tsn.append(r.test.snr); sc.append(r.scale)
    time.sleep(0.15)
picam2.stop()

print("control detected %d/%d  snr mean %.1f max %.1f" % (cdet, N, sum(csn)/N, max(csn)))
print("test    detected %d/%d  snr mean %.1f max %.1f" % (tdet, N, sum(tsn)/N, max(tsn)))
print("profile scale mean %.2f" % (sum(sc)/N))

warped = strip.warp_quad(last, np.asarray(cal.window_quad, np.float32),
                         strip.WINDOW_CANON_W, strip.WINDOW_CANON_H)
roi = strip.extract_strip_roi(warped)
gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
print("strip brightness: mean %.0f max %d   pct>250: %.1f%%"
      % (gray.mean(), gray.max(), 100 * (gray > 250).mean()))
