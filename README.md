# Lateral Flow Reader

Quantitative reader for lateral flow tests (COVID-19 rapid antigen tests and
similar cassettes), built on a Raspberry Pi with Picamera2 and OpenCV.

Rather than just calling a test positive or negative, it watches the cassette
for the whole development period and measures how the test line *develops*.

## What it measures

| | |
|---|---|
| **Time to positivity** | when the test line first became reliably visible |
| **Band density** | how much dye ended up on the test line, as a viral load proxy |
| **Rate of change** | how fast the line darkened, from first detection to plateau |

Density is reported as integrated band area in raw LAB `a*` units, and as a
**T/C area ratio** — normalising the test line by the control absorbs
run-to-run variation in sample volume and flow rate, so that ratio is the
figure to compare between cassettes.

## How it works

```
camera frame
  -> locate cassette (Canny + contour, largest convex quad)
  -> perspective warp to a canonical 900x320 view
  -> slice results window, then the strip ROI (fixed fractions of the warp)
  -> LAB a* channel, Gaussian background subtraction, vertical median
     -> 1-D redness profile
  -> peak detection, band-width filtering, T/C assignment by expected position
  -> SNR gating, then temporal voting across frames
```

Working in a canonical warped space means every region of interest is a
fraction rather than a pixel count, so the reader tolerates the camera moving.

Two properties are load-bearing and worth knowing before changing anything:

- **Detection and density use different profiles.** Detection runs on a
  noise-normalised profile, where SNR thresholds are meaningful. Density is
  measured on the raw `a*` profile, because the normalising divisor grows
  with the band itself and compresses the dose response. Storing the scale
  factor keeps both recoverable from one recording.
- **Rendering never touches measured data.** `strip.analyze()` returns
  measurements; `viz.render()` draws on copies. Overlays previously bled into
  the pixels being measured and inflated SNR by roughly 45%.

## Layout

| file | role |
|---|---|
| `main.py` | capture loop, CSV logging, run lifecycle |
| `strip.py` | cassette geometry and all signal analysis (no camera, no display) |
| `viz.py` | display overlays (draws only on copies) |
| `recorder.py` | per-run capture of profiles and decimated frames |
| `analysis.py` | offline analysis of a recorded run |
| `tests/` | pytest suite, runs on a workstation |

`strip.py`, `viz.py`, `recorder.py` and `analysis.py` import nothing
Pi-specific, so everything except the capture loop can be developed and
tested on a laptop.

## Install

On Raspberry Pi OS, **picamera2 comes from apt, not pip** — it needs libcamera
bindings pip cannot supply. It is preinstalled on Bookworm; otherwise:

```sh
sudo apt install -y python3-picamera2
```

Then the rest:

```sh
pip install -r requirements.txt
```

If you prefer a virtualenv, create it with `--system-site-packages` or the
apt-installed picamera2 will not be visible inside it:

```sh
python3 -m venv --system-site-packages .venv
```

## Running a test

```sh
python3 main.py
```

Point the camera at the cassette. The reader searches for it every frame and
starts measuring as soon as it locks on. Live windows show the camera view
with detected bands projected back onto it, the warped cassette, the strip
ROI, and the 1-D signal profile.

A run stops automatically after `RUN_DURATION_S` (15 minutes) so that runs are
comparable and the plateau is well defined. `q` stops early; the summary line
records which happened.

Each run writes to its own timestamped directory — restarting the reader never
overwrites previous data:

```
runs/20260722-113353/
    signal_log.csv    per-frame measurements
    profiles.npz      per-frame 1-D profiles and band readings
    frames/           warped cassette images, every 2 s
```

## Analysing a run

```sh
python3 analysis.py                        # newest run under runs/
python3 analysis.py runs/20260722-113353
python3 analysis.py --test-snr 8 --json
```

```
Time to positivity
  onset      180.0 s
  confirmed  180.6 s  (7 of 10 frames)

Band density (raw a* units, plateau = median of run tail)
  T area          200.00 a*.px
  T peak          17.39 a*
  T/C area ratio  1.0000   <- compare across runs

Rate of change (onset -> plateau)
  interval        180.0 -> 582.5 s  (402.5 s)
  T area rate     0.4762 a*.px/s   (linear fit R^2 1.000)
  endpoint rate   0.4747 a*.px/s
  T/C ratio rate  0.002381 /s   (R^2 1.000)
```

Detection is re-derived from the stored SNRs rather than the recorded verdict,
so `--test-snr` and `--control-snr` re-score a finished run at new thresholds.
A cassette is one-shot, so tuning happens against recordings, not fresh tests.

Two reported figures are worth reading carefully. `onset` is back-dated to the
first detection in the window that satisfied the vote threshold, since the
latch necessarily fires later than the line appears — `confirmed` is when it
latched. And the rate comes with an R²: a low value means a single slope is a
poor description of the rise, which is normal for a saturating curve.

## Validity

A test line is only reported when the control line is also present, and a band
at the test position is never accepted as the control. A cassette whose
control never developed is reported as **invalid** rather than as a negative —
failing toward "unreadable" instead of toward a false negative.

## Tuning

Detection constants live at the top of `strip.py`:

| constant | meaning |
|---|---|
| `EXPECTED_T_FRAC`, `EXPECTED_C_FRAC` | where the bands sit along the strip |
| `SEARCH_RADIUS_FRAC` | how far from there to look |
| `MIN_BAND_WIDTH`, `MAX_BAND_WIDTH` | plausible band widths, in profile samples |
| `TEST_SNR_THRESHOLD`, `CONTROL_SNR_THRESHOLD` | detection thresholds |
| `STABILITY_WINDOW`, `STABILITY_VOTES` | how many frames a detection must hold for |

Geometry (`WINDOW_*`, `STRIP_*`) is expressed as fractions of the canonical
warp and will need adjusting for a cassette of a different shape.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

The suite stubs picamera2 and libcamera, so it needs no Pi and no display. It
works on synthetic cassettes whose kinetics are known by construction, which
is how the analysis figures are checked against ground truth.

## Status

Validated end-to-end against synthetic data. Detection thresholds still need
calibrating against real cassettes — record a run, then sweep thresholds
offline with `analysis.py --test-snr`.

Known limitation: the cassette quad is re-detected independently every frame,
so the ROI jitters by a pixel or two between frames. That noise lands in the
series used for the rate-of-change measurement.
