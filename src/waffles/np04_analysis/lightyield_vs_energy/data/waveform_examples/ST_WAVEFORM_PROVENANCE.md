# Self-trigger waveform illustration: data provenance

These files support an illustration of waveform processing. They are **not**
the original recorded ADC samples of a verified self-trigger event.

- `processed_st_screenshot.png` is the user-supplied image of a single
  processed waveform. The image displays approximately ticks 0–400.
- `st_waveform_after_digitized.csv` contains one interpolated amplitude per
  displayed tick. Pixel positions were calibrated from the visible axes of
  the screenshot. Values are approximate.
- `st_waveform_before_reconstructed.csv` was calculated as
  `8220 ADC - digitized processed amplitude`. The 8220 ADC baseline is an
  illustrative assumption; it was not measured for this waveform.

The two CSVs have columns `time_tick,amplitude_adc`. Run number, channel,
acquisition mode, and the remaining samples of a full 1024-sample ST
record cannot be established from this screenshot. These values must not be
used as event data in quantitative analyses. Any thesis use should identify
the panels as an illustrative reconstruction from a screenshot.
