from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

p = Path("demo_data")
p.mkdir(exist_ok=True)
crs = CRS.from_string("+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs")
ys, xs = np.mgrid[0:128, 0:256]
img = (0.3 + 0.4 * np.exp(-((ys - 64) ** 2 + (xs - 80) ** 2) / 800) * (0.5 + 0.5 * np.cos(xs * 0.05))).astype("float32")

label_tmc = """<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <logical_identifier>urn:isro:ch2:tmc:TMC_demo</logical_identifier>
  <start_date_time>2019-09-12T00:00:00Z</start_date_time>
  <incidence_angle>42</incidence_angle>
  <emission_angle>3</emission_angle>
  <phase_angle>48</phase_angle>
  <solar_azimuth>112</solar_azimuth>
  <pixel_resolution>5.0</pixel_resolution>
  <west_bounding_coordinate>-0.07</west_bounding_coordinate>
  <east_bounding_coordinate>0.07</east_bounding_coordinate>
  <south_bounding_coordinate>0.0</south_bounding_coordinate>
  <north_bounding_coordinate>0.04</north_bounding_coordinate>
</Product_Observational>
"""
label_ohrc = label_tmc.replace("tmc:TMC_demo", "ohrc:OHRC_demo").replace(">5.0<", ">0.32<")

with rasterio.open(
    p / "TMC_demo.tif",
    "w",
    driver="GTiff",
    height=128,
    width=256,
    count=1,
    dtype="float32",
    crs=crs,
    transform=from_origin(-2000, 3000, 5, 5),
) as dst:
    dst.write(img, 1)
(p / "TMC_demo.xml").write_text(label_tmc, encoding="utf-8")

with rasterio.open(
    p / "OHRC_demo.tif",
    "w",
    driver="GTiff",
    height=128,
    width=256,
    count=1,
    dtype="float32",
    crs=crs,
    transform=from_origin(-2000, 3000, 0.32, 0.32),
) as dst:
    dst.write((img * 1.2).astype("float32"), 1)
(p / "OHRC_demo.xml").write_text(label_ohrc, encoding="utf-8")
print("wrote", p)
