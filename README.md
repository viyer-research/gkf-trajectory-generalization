# Graph Kalman Filtering: Battery-Gated Sensor Validity and Trajectory Generalization

Source for the ICST-2026 submission *"Battery-Gated Measurement Validity in
Wireless Sensor Deployments: Fault Characterisation, Recovery, and Downstream
Consequences"*, plus a follow-up generalization study applying the same
graph-message-passing Kalman filter to human GPS mobility trajectories
(TrajImpute ETH-M/HOTEL-M and GeoLife).

## Structure

```
paper/                          LaTeX source for the ICST-2026 submission
  icst_measurement_validity.tex
  icst_measurement_validity.pdf
  IEEEtran.cls
  figs/                         figures referenced by the paper

generalization/                 external-validity study (Section X of the paper)
  README.md                     full write-up: setup, results, caveats, bugs found/fixed
  eth_hotel/                    TrajImpute ETH-M / HOTEL-M pedestrian benchmarks
    scripts/
      torchless_load.py         pure-Python loader for TrajImpute's torch-pickled .pkl files
      gkf_trajimpute.py         fits theta_tm/theta_sp, runs KF + persistence baseline,
                                 evaluates Easy/Hard protocols (set GKF_DATASET=ETH-M|HOTEL-M)
    results_ETH-M.json
    results_HOTEL-M.json
  geolife/                      GeoLife Beijing GPS trajectories, built from scratch
    scripts/
      geolife_prepare.py        stage 1: read raw GPS logs from the GeoLife zip, tag by
                                 transportation mode, restrict to a bounded scope
      geolife_windows.py        stage 2: resample onto a global 5s grid, cut 8+12-step windows
      geolife_split.py          stage 3: build co-presence scenes, simulate Easy/Hard
                                 missingness, user-disjoint 70/15/15 split, per mode
      gkf_geolife.py            stage 4: fit + evaluate per (mode, protocol), incl. the
                                 per-scene coordinate-centring fix (see generalization/README.md)
    results_GeoLife.json
    results_GeoLife_by_m.json   error broken down by exact missing-count m
```

## Reproducing

Dependencies: Python 3.10+, `numpy` only (see `requirements.txt`). No `torch`
install is required — `torchless_load.py` re-implements just enough of
torch's legacy pickle/storage protocol to read TrajImpute's `.pkl` files.

Raw data is not included in this repository (large, and not ours to
redistribute):

- **ETH-M / HOTEL-M**: from TrajImpute (Chib & Singh, NeurIPS 2024
  Datasets & Benchmarks Track), <https://github.com/Pranav-chib/TrajImpute>.
- **GeoLife**: Microsoft Research GeoLife GPS Trajectories dataset
  (Zheng, Xie & Ma, 2010), publicly downloadable from Microsoft Research.

Point `GKF_DATASET` / `DATA_ROOT` (eth_hotel) or `ZIP_PATH` (geolife,
`geolife_prepare.py`) at your local copies and run the stage scripts in
order; each stage writes intermediate `.npz`/`.pkl`-derived artifacts the
next stage reads.

## Paper ↔ code mapping

The paper's Data Availability section lists which script backs which
section. In summary: Sections IV–IX (fault characterisation, fusion,
staged cleaner, downstream KF, recovery, locality) use the Intel Berkeley
Research Lab deployment and are not included in this repo (separate
pipeline); Section X (External Validity) uses `generalization/`.

## License

TBD — data licenses (Intel Lab data, TrajImpute, GeoLife) apply
independently to their respective raw datasets, none of which are
redistributed here.
