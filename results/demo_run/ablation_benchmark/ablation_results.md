| Method | Inlier Count | Inlier Ratio | Pixel RMSE | Absolute RMSE (m) | Processing Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure SIFT** | 4 | 57.1% | inf px | 0.00 m | 0.13 s |
| **Pure LoFTR (Fallback: NCC)** | 0 | 0.0% | N/A | N/A | 0.06 s |
| **Our Pipeline (No DEM)** | 0 | 0.0% | N/A | N/A | 1.76 s |
| **Our Full Pipeline (CFOG+DEM+Grid NMS)** | 0 | 0.0% | N/A | N/A | 1.30 s |