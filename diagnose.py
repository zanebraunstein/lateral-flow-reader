import time
import cv2 as cv
import calibration as calib
import strip
import viz
import main

picam2 = main.start_camera()
time.sleep(1.5)
cal = calib.load()
frame = main.grab_frame(picam2)
picam2.stop()

r = strip.analyze(frame, cal.window_quad, cal.bands)
n = len(r.profile)
frac = lambda i: None if i is None else round(i / n, 3)

# the actual strip the profile is computed from, upscaled 3x
big = cv.resize(r.strip_roi, (r.strip_roi.shape[1] * 3, r.strip_roi.shape[0] * 3),
                interpolation=cv.INTER_NEAREST)
cv.imwrite("diag_strip.png", big)

# the profile with T (cyan), C (green), candidates (yellow) marked
pv = viz.draw_profile(r.profile)
viz.mark_bands_on_profile(pv, r.profile, r.test.idx, r.control.idx, r.candidates)
cv.imwrite("diag_profile.png", pv)

print("saved diag_strip.png and diag_profile.png")
print("calib:    control=%.3f  test=%.3f" % (cal.control_frac, cal.test_frac))
print("detected: control=%s (snr %.1f)  test=%s (snr %.1f)"
      % (frac(r.control.idx), r.control.snr, frac(r.test.idx), r.test.snr))
print("candidates:", [(round(c / n, 3), round(float(r.profile[c]), 1)) for c in r.candidates])
