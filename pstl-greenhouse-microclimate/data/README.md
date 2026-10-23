# Winter input data

These three CSV files are the inputs used by the standalone experiment package. `smoothed_winter_indoor_environment.csv` contains indoor sensor channels; `winter_outdoorenv_solardata.csv` contains outdoor meteorology and solar-related columns; `winter_outdoor_data.csv` contains outdoor meteorology without the solar-derived feature set. The code aligns timestamps and derives cyclic solar geometry from the configured site and timestamp.

The original inputs were copied without transformation. Verify integrity with `data_manifest.json`. Spatial node mapping is documented separately in `../configs/spatial/sensor_mapping.yaml`; its relative coordinates are figure-derived, not surveyed.
