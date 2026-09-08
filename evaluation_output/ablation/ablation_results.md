| Method | Inlier Count | Inlier Ratio | Pixel RMSE | Absolute RMSE (m) | Processing Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure SIFT** | 4 | 57.1% | 0.000 px | 0.00 m | 0.15 s |
| **Pure LoFTR (NCC Fallback)** | 0 | 0.0% | N/A | N/A | 0.05 s |
| **Our Pipeline (No DEM)** | 6 | 7.8% | 1.325 px | 7.16 m | 1.92 s |
| **Our Full Pipeline (CFOG+DEM+Grid NMS)** | 6 | 7.8% | 1.325 px | 7.16 m | 1.21 s |