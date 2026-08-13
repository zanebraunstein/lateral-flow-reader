import time
import calibration as calib
import strip
import main

picam2 = main.start_camera()
time.sleep(1.0)
cal = calib.load()
frame = main.grab_frame(picam2)
picam2.stop()

r = strip.analyze(frame, cal.window_quad, cal.bands)
n = len(r.profile)
frac = lambda i: None if i is None else round(i / n, 3)

print("calibration: control_frac=%.3f test_frac=%.3f" % (cal.control_frac, cal.test_frac))
print("candidates (position, height, width):")
for c in r.candidates:
    print("   frac=%.3f  height=%.2f  width=%d"
          % (c / n, float(r.profile[c]), strip.band_width_peak(r.profile, c)))
print("CONTROL detected at frac=%s  snr=%.1f" % (frac(r.control.idx), r.control.snr))
print("TEST    detected at frac=%s  snr=%.1f" % (frac(r.test.idx), r.test.snr))
print("limits: MIN_W=%d MAX_W=%d edge=%.2f radius=%.2f"
      % (strip.MIN_BAND_WIDTH, strip.MAX_BAND_WIDTH,
         strip.EDGE_EXCLUDE_FRAC, strip.SEARCH_RADIUS_FRAC))
