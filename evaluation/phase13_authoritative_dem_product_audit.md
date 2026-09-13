# Phase 13 — Authoritative LOLA / SLDEM2015 Product Identification & Provenance Audit

**Date:** September 13, 2026
**Status:** Complete — Research + Provenance Audit Only
**Classification:** **BLOCKED_BOTH**
**Production Integrity:** Frozen and Unmodified (`half_p=8`, `use_goa_preselection=False`, `ML_model/ai_verifier_model.pkl` unchanged, `ML_model/ground_truth_matches.json` unchanged)

> No DEM ingestion implemented. No production matcher, RF model, candidate-generation defaults, ground-truth, or correspondence experiments modified. `dem_512.png` NOT used as elevation data. No product ID, URL, footprint, or solar angle fabricated.

---

## A. Executive conclusion

Phase 12 concluded `BLOCKED`: footprints known, hillshade engine ready, but authoritative altimetry product IDs and solar-elevation metadata were not identified in-repo.

Phase 13 resolves the **product-identification** half:

- Both target footprints lie inside a **single PDS dataset**: `LRO-L-LOLA-4-GDR-V1.0` (NASA PDS Geosciences Node, producer LRO LOLA team, GSFC).
- **One LOLA family** (`LDEM_512` cylindrical, FLOAT_IMG preferred) and **one SLDEM2015 family** (512 ppd TILES + 256/128 GLOBAL) cover **all six regions**. Exact tiles enumerated in §E.
- **Recommended source: SLDEM2015 512 ppd tiles** (§J) — ~60 m posting like LOLA 512, but gap-filled with co-registered Kaguya TC so no interpolation is needed on these small equatorial tiles; GIS-ready JP2 + AUX.XML reads directly in rasterio/GDAL.
- **Still blocked on external file supply + solar metadata** (§K, §N): the authoritative rasters and the Chandrayaan-2 PDS4 XML labels for regions 002–006 are not in the repository and were not downloaded (would require ISRO PRADAN registration and explicit operator supply with checksums). Solar elevation for regions 002–006 remains unknown and is NOT inferred.

Final decision: **BLOCKED_BOTH** — products identified, files + solar labels still missing.

---

## B. Region footprints

Source: `data_preprocessing_pipeline/processed_triplets/region_*/manifest.json` (read-only inspection). CRS used by tiling code (`make_demo_regions.py`): lunar geographic `+proj=longlat +a=1737400 +b=1737400` and Moon Equirectangular `+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m`. Task statement rounds these to Lon 336.48–336.59 E / Lat −3.38–−2.74 N (001–004, Sinus Medii) and Lon 234.39–234.53 E / Lat +4.86–+5.35 N (005–006, equatorial highland); manifests agree to 1e-4 deg.

| Region | west_lon | east_lon | south_lat | north_lat | OHRC product | TMC-2 product | Azimuth OHRC / TMC-2 (manifest) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| region_001 | 336.484646 | 336.589455 | −3.3748612 | −3.2487328 | `ch2_ohr_ncp_20210405t1606536730_d_img_d18` | `ch2_tmc_ncf_20250807t1904346039_d_img_d18` | 269.646098 / 108.866212 |
| region_002 | 336.484646 | 336.589455 | −3.2066900 | −3.0805616 | same as 001 | same as 001 | same as 001 |
| region_003 | 336.484646 | 336.589455 | −3.0385188 | −2.9123904 | same as 001 | same as 001 | same as 001 |
| region_004 | 336.484646 | 336.589455 | −2.8703476 | −2.7442192 | same as 001 | same as 001 | same as 001 |
| region_005 | 234.396774 | 234.528638 | +4.8631734 | +5.0672006 | `ch2_ohr_ncp_20220914t0835371412_d_img_d32` | `ch2_tmc_ncf_20191125t0749024661_d_img_d18` | 89.392682 / 251.647896 |
| region_006 | 234.396774 | 234.528638 | +5.1488115 | +5.3528388 | same as 005 | same as 005 | same as 005 |

Source-image identifiers (from manifests + `make_demo_regions.py` bundles + `triplets.json`):

- Bundle 1 (regions 001–004): OHRC `ch2_ohr_ncp_20210405T1606536730_d_img_d18`, TMC-2 `ch2_tmc_ncf_20250807T1904346039_d_img_d18`, IIRS `ch2_iir_nri_20211221T0324126144_d_img_hw1`.
- Bundle 2 (regions 005–006): OHRC `ch2_ohr_ncp_20220914T0835371412_d_img_d32`, TMC-2 `ch2_tmc_ncf_20191125T0749024661_d_img_d18`, IIRS `ch2_iir_nri_20220221T1109265965_d_img_d18`.
- Filename convention (USGS Astrogeology, confirms PDS4): `ch2_<ohr|tmc>_<ncp|ncf>_YYYYMMDDTHHMMSSssss_d_img_<d18|d32>` + detached `.xml` label; calibrated products carry `c` in the phase segment (`ncp`/`ncf`/`ncn`/`nca`).

Existing repo capabilities (read-only):

- `backend/data/lro_validator.py` (`LRODataHandler`): point sampler only — opens `.tif`/rasterio-readable raster, returns array + transform/CRS/bounds, samples one elevation per lat/lon. No crop/warp/resample/download, no 2-D window extraction.
- `ML_model/lro_ode_client.py`: hardcoded to LRO NAC optical (`ihid="lro"`, `iid="lroc"`, `pt="EDRNAC"`, ODE `https://oderest.rsl.wustl.edu/live2/`), decodes PDS3 optical frames via `decode_pds3_img_to_array`. No LOLA/GDR or GeoTIFF/JP2 altimetry path.
- `data/ingestion/pds4_reader.py`: parses PDS4 XML (`geom:sun_azimuth`, `geom:sun_elevation`, GSD, dimensions, SPICE refs) and legacy VICAR; falls back to incidence-derived elevation (`elev = 90 − incidence`) only when the tag exists. Fully functional parser, but labels for 002–006 are absent from git.
- `ML_model/matcher_cfog.py` L1070–L1255: `compute_dem_cast_shadows` (ray-marched shadows) + `render_synthetic_shaded_relief` (Horn slope/aspect, Lambertian) + `sun_azimuth_delta_deg` / `resolve_sun_elevation_deg` (incidence fallback, 45° default). Idle without calibrated DEM input.
- `make_demo_regions.py` L163–180: `dem_512.png` is **synthesized from TMC-2 optical gradients/stereo parallax** (`cv2.Sobel`, `derive_dem_from_tmc_stereo`) — explicitly NOT authoritative elevation and NOT used in this audit.

---

## C. Candidate authoritative LOLA products

PDS dataset: **`LRO-L-LOLA-4-GDR-V1.0`**, "LRO Moon Laser Altimeter 4 GDR V1.0", NASA PDS Geosciences Node. Producer: Gregory A. Neumann / David E. Smith, LRO LOLA team, Goddard Space Flight Center. Reference SIS: LOLA RDR/GDR/SHADR SIS (archsis.pdf). DOI-style citation: Neumann, G.A., 2009, LRO-L-LOLA-4-GDR-V1.0, NASA PDS, 2010.

- Product family: **`LDEM_512`** cylindrical (Simple Cylindrical / Equirectangular), 512 pixels/deg → **59.225 m/px in latitude** (~59 m at equator).
- Tiling: longitudes `000_090 / 090_180 / 180_270 / 270_360` × latitude bands `90S_45S / 45S_00S / 00N_45N / 45N_90N` (16 tiles at 512 ppd); also global `LDEM_4/16/64/128` and `LDEM_256` (4 tiles) and `LDEM_1024` (30°×15° tiles, sparse/preliminary).
- File variants per tile: PDS integer `IMG+LBL` (16-bit, `SCALING_FACTOR=0.5`, `OFFSET=1737400`, i.e. `radius = DN×0.5 + 1737400`), 32-bit float `FLOAT_IMG` (`*_FLOAT.IMG + *.LBL + *.XML`, elevation directly in meters), and `JP2` GeoJPEG2000 with geospatial headers.
- Datum/frame: radius values relative to **1737.4 km reference sphere** (`A=B=C=1737.4 km`); cylindrical labels state `COORDINATE_SYSTEM_TYPE="BODY-FIXED ROTATING"`, `COORDINATE_SYSTEM_NAME="MEAN EARTH/POLAR AXIS OF DE421"` (plus a `LDEM_64_PA` Principal-Axis variant for gravity consistency — NOT recommended here to avoid a ~km-level frame rotation vs. the tiling CRS).
- Longitude: **0–360 East** (`WESTERNMOST_LONGITUDE=0`, `EASTERNMOST_LONGITUDE=360`, `POSITIVE_LONGITUDE_DIRECTION=EAST`); latitude planetocentric −90…+90.
- Recommended LOLA pick for Phase 14: **`LDEM_512` `FLOAT_IMG` tiles** (float32 meters, no scale math, GDAL/rasterio-readable; PDS3 label + PDS4 XML sidecars present in `float_img/`).
- Calibrated/derived: yes — binned, crossover-adjusted, interpolated altimetry DEM (CODMAC Level 4 GDR), suitable terrain data for hillshade synthesis, with the caveat of interpolated gaps between equatorial ground tracks.

Authoritative locations:

- Landing/profile: `https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LRO-L-LOLA-4-GDR-V1.0`
- PDS Geosciences mirror (exact tile filenames verified in directory listing): `https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/`
- LOLA PDS node + docs: `https://imbrium.mit.edu/`, `https://imbrium.mit.edu/DOCUMENT/archsis.pdf`, ODE help `https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/LOLA/GDR/GDRDEM.htm`
- Reference: Smith et al. 2009 (LOLA instrument); Lemoine et al. 2014 (GRGM900C orbits for current labels).

---

## D. Candidate authoritative SLDEM2015 products

Same PDS dataset **`LRO-L-LOLA-4-GDR-V1.0`** (directories `SLDEM*`), produced by Barker et al. (NASA GSFC), merging ~4.5×10⁹ LOLA heights with 43,200 co-registered SELENE (Kaguya) Terrain Camera 1°×1° stereo DEMs. Reference: Barker et al., Icarus 2016, doi:10.1016/j.icarus.2015.07.039 ("A new lunar digital elevation model from LOLA and SELENE TC").

- Product family: **`SLDEM2015`**, 512 ppd (~60 m at equator), coverage ±60° latitude × 360° longitude.
- Tiling at 512 ppd: **30° latitude × 45° longitude** tiles, e.g. `SLDEM2015_512_30S_00S_315_360`, `SLDEM2015_512_00N_30N_225_270` (verified naming pattern in `https://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/` index). GLOBAL 128/256 ppd single-file products also exist (`SLDEM2015_128_60S_60N_000_360`, `SLDEM2015_256_60S_60N_000_360`) plus `SLDEM2015_DATA_QUALITY_*`.
- Formats per tile: PDS float `IMG+LBL` **and** `JP2 + AUX.XML + JP2.LBL` (GeoJPEG2000 with auxiliary georeferencing — directly GIS/rasterio-ready). Document note (`sldem2015.txt`): GLOBAL in `GLOBAL/`, 512 ppd in `TILES/`.
- Datum/frame: LOLA GRAIL-based geodetic framework (co-registered TC); same 1737.4 km sphere convention; cylindrical, 0–360 E, ±60°.
- Accuracy (Barker et al.): bulk TC-vs-LOLA RMS residual 3–4 m after co-registration (~90% of tiles <5 m vs ~50% before); LOLA absolute ~1 m (Mazarico et al. 2013); adopted SLDEM2015 typical vertical accuracy **3–4 m**; profile corrections typically <10 m horizontal, <1 m vertical.
- Calibrated/derived: yes — merged terrain DEM designed to fill LOLA inter-track gaps without surface interpolation; stated PGDA purpose includes orthorectification/co-registration support.

Authoritative locations:

- PGDA product page: `https://pgda.gsfc.nasa.gov/products/54`
- LOLA node SLDEM2015 browser + tile index: `https://imbrium.mit.edu/BROWSE/SLDEM2015/TILES/`, `https://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/`
- ODE SLDEM help: `https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/LOLA/SLDEM.htm`
- Archive note: `https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/document/sldem2015.txt`

---

## E. Coverage verification

Both footprints are near-equatorial, well inside the ±60° SLDEM2015 limit and inside global LOLA coverage. Tile containment was checked against the published tiling schemes (LDEM_512: 90° lon × 45° lat; SLDEM2015_512: 45° lon × 30° lat) — no coverage assumed from names alone; tile ranges containing the manifest bounds are listed.

| Product (exact tile/file stem) | Coverage | Resolution | Format | CRS | Units | Region coverage | Authoritative source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `LDEM_512_45S_00S_270_360_FLOAT` (`ldem_512_45s_00s_270_360_float.img/.lbl/.xml`) | lon 270–360 E, lat 45S–00S | 512 ppd, 59.225 m/px | PDS float32 IMG + LBL + PDS4 XML | Simple Cylindrical, 0–360 E, ME/PA DE421, R=1737.4 km | meters rel. 1737400 | region_001, 002, 003, 004 (lon 336.48–336.59, lat −3.37–−2.74 all inside) | PDS Geosciences `float_img/` + ODE GDRDEM help |
| `LDEM_512_00N_45N_180_270_FLOAT` (`ldem_512_00n_45n_180_270_float.img/.lbl/.xml`) | lon 180–270 E, lat 00N–45N | 512 ppd, 59.225 m/px | PDS float32 IMG + LBL + PDS4 XML | Simple Cylindrical, 0–360 E, ME/PA DE421, R=1737.4 km | meters rel. 1737400 | region_005 (234.39–234.53 E, 4.86–5.07 N), region_006 (234.39–234.53 E, 5.15–5.35 N) | PDS Geosciences `float_img/` + `imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/FLOAT_IMG/` |
| `SLDEM2015_512_30S_00S_315_360` (`.JP2` + `_AUX.XML` + `_JP2.LBL`; IMG variant also archived) | lon 315–360 E, lat 30S–00S | 512 ppd, ~60 m/px | GeoJPEG2000 + AUX.XML (GIS-ready); PDS IMG mirror | Cylindrical, 0–360 E, GRAIL/LOLA framework | meters terrain height | region_001–004 (same bounds as above) | `imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/` + `sldem2015.txt` |
| `SLDEM2015_512_00N_30N_225_270` (`.JP2` + `_AUX.XML` + `_JP2.LBL`; IMG variant also archived) | lon 225–270 E, lat 00N–30N | 512 ppd, ~60 m/px | GeoJPEG2000 + AUX.XML (GIS-ready); PDS IMG mirror | Cylindrical, 0–360 E, GRAIL/LOLA framework | meters terrain height | region_005, region_006 (lon 234.39–234.53 inside 225–270; lat 4.86–5.35 inside 00N–30N) | `imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2/` + ODE SLDEM help |
| `SLDEM2015_256_60S_60N_000_360` / `SLDEM2015_128_60S_60N_000_360` (GLOBAL) | ±60° × 360° global | 256 ppd (~120 m) / 128 ppd (~240 m) | IMG + JP2 | Cylindrical | meters | all six regions (lower-res fallback only) | ODE SLDEM help + `sldem2015.txt` |

Result: **two tiles per family cover all six regions** (one tile per footprint cluster). No polar products needed. If a single-file option is required, the GLOBAL SLDEM2015 256/128 or global LDEM_64/128 cover everything at reduced resolution — not recommended for hillshade work.

---

## F. Product specifications

| Attribute | LOLA `LDEM_512` (recommended FLOAT_IMG) | SLDEM2015 512 ppd tiles (recommended) |
| --- | --- | --- |
| PDS dataset | `LRO-L-LOLA-4-GDR-V1.0` | `LRO-L-LOLA-4-GDR-V1.0` (SLDEM* dirs) |
| Version | Product labels e.g. `PRODUCT_VERSION_ID V3.0`, `PRODUCT_CREATION_TIME 2017-09-15` (LDEM_128 reference label); LOLA releases ongoing (e.g. Release 37 revises 16/64 ppd) — record exact `.LBL`/`.XML` on download | 2015 archive (`*.JP2` Jul 2015, `*_JP2.LBL` Aug 2015 in tile index); reference Barker et al. 2015/2016 |
| Raster resolution | 512 ppd = 59.225 m/px lat (≈59 m at equator) | 512 ppd ≈ 60 m/px at equator |
| Elevation units | meters relative to 1737400.0 (float IMG direct; int IMG via `radius = DN×0.5 + 1737400`) | meters terrain height (same LOLA framework) |
| Projection/CRS | Simple Cylindrical (Equirectangular), `CENTER_LONGITUDE=180`, `MAP_SCALE=59.225` at 512 | Cylindrical (Equirectangular-equivalent), same family |
| Longitude convention | 0–360 East | 0–360 East |
| Latitude convention | Planetocentric −90…+90 | Planetocentric, product limit ±60° (both footprints inside) |
| Vertical datum | Sphere R=1737.4 km (`OFFSET=1737400`, `A=B=C=1737.4`); topography = radius − geoid, geoid≈OFFSET at equator | Same LOLA/GRAIL geodetic framework (TC co-registered to it) |
| File format | `*_FLOAT.IMG` + `.LBL` + `.XML` (preferred); integer `.IMG` + `.LBL`; `.JP2` mirrors | `.JP2` + `_AUX.XML` + `_JP2.LBL` (preferred GIS path); `.IMG` mirror |
| Globally tiled or regional | Global in tiles (512: 90°×45° tiles) | Near-global ±60° in tiles (512: 45°×30° tiles) + GLOBAL 128/256 |
| GeoTIFF directly? | No native GeoTIFF; JP2+AUX opens in GDAL/rasterio; FLOAT_IMG reads via rasterio with PDS driver or trivial `numpy.fromfile` + label geometry | **Yes effectively**: JP2 + AUX.XML is the GIS-ready path; opens in rasterio/GDAL/QGIS |
| Calibrated terrain for hillshade? | Yes (gridded DEM), with interpolated inter-track gaps at equator | Yes — preferred at equator because TC fills inter-track gaps without interpolation |

Representative exact filenames (stems; extensions in §E):

- LOLA: `ldem_512_45s_00s_270_360_float`, `ldem_512_00n_45n_180_270_float`
- SLDEM2015: `SLDEM2015_512_30S_00S_315_360`, `SLDEM2015_512_00N_30N_225_270`

---

## G. Solar-geometry metadata findings

Repo-first inspection:

- `processed_triplets/region_*/manifest.json`: `ohrc_sun_azimuth_deg` + `tmc2_sun_azimuth_deg` present for all six regions (001–004: 269.646098/108.866212; 005–006: 89.392682/251.647896). **No `sun_elevation` keys in any manifest.**
- `data_preprocessing_pipeline/triplets.json` (second entry, the real 2021/2025 triplet): OHRC az 269.646098 / **elev 14.063051**, TMC-2 az 108.866212 / **elev 48.753985**, IIRS az 335.798629 / elev 82.736335, with `ohrc_label`/`tmc2_label` pointing to offline `<external-downloads-dir>` XML paths. This is the **sole in-repo source** for the historical region_001 values cited in Phase 12 (≈14.06° OHRC, ≈48.75° TMC-2).
- `pds4_reader.py` can extract `sun_azimuth`/`sun_elevation` (or derive elevation as 90−incidence when tagged) plus observation time, SPICE refs, footprint, and image linkage — but **the XML files it needs are not in git** for any region 002–006 product.
- `matcher_cfog.py::resolve_sun_elevation_deg` falls back incidence→elevation→45° default. Per task constraints this fallback must NOT be used as a substitute for measured solar elevation in Phase 14.

Verification status:

- region_001 historical values (14.063051° / 48.753985°) **confirmed present in `triplets.json`** — treated strictly as historical repository evidence, NOT as authoritative label recovery, because the underlying XML is offline.
- regions 002–006: solar elevation **missing** (manifests carry azimuth only; `triplets.json` has no second-triplet rows for the 2022/2019 bundle). Azimuth exists, so hillshade azimuth geometry is known, but elevation is unknown.
- Spacecraft position / observation time / target-body / footprint: recoverable only from the missing PDS4 labels + SPICE kernels (none committed). No `*.bsp/*.bc/*.ti` in repo.

Nothing inferred from azimuth, images, or defaults in this audit.

---

## H. PDS4 label recovery requirements

Required Chandrayaan-2 calibrated PDS4 detached labels (`.xml` + sibling `.img` in same directory for ISIS/`pds4_reader` use):

| Region(s) | Required OHRC label stem | Required TMC-2 label stem | Archive |
| --- | --- | --- | --- |
| 001–004 | `ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml` | `ch2_tmc_ncf_20250807T1904346039_d_img_d18.xml` | ISRO PRADAN `https://pradan.issdc.gov.in/ch2/` / map browser `https://chmapbrowse.issdc.gov.in/` (registration required) |
| 005–006 | `ch2_ohr_ncp_20220914T0835371412_d_img_d32.xml` | `ch2_tmc_ncf_20191125T0749024661_d_img_d18.xml` | same PRADAN archive |
| supporting | IIRS `ch2_iir_nri_20211221T0324126144_d_img_hw1.xml` (001–004), `ch2_iir_nri_20220221T1109265965_d_img_d18.xml` (005–006) | — | same PRADAN archive |

Sufficiency: yes — the `ch2_<inst>_<phase>_YYYYMMDDTHHMMSSssss_d_img_<station>` stems in manifests follow the documented ISRO/USGS convention and are directly searchable in PRADAN / the USGS `CH2_OHRC_Calibrated_Product` / `CH2_TMC_Calibrated_Product` layers (ISIS docs confirm `isisimport from=*.xml`, station codes `d18/d32`, calibrated `ncp/ncf`). No download attempted: PRADAN requires login/registration, bulk fetch is ambiguous without operator credentials, and the task forbids automatic ingestion. The operator must supply the six XMLs (optionally + IIRS) with SHA-256.

Each label should provide: `geom:sun_azimuth`, `geom:sun_elevation` (or incidence/emission/phase triple), observation start/stop UTC, spacecraft position/pointing or SPICE kernel refs, target `Moon`, footprint polygon, and `file_name` linkage. USGS confirms OHRC/TMC-2 calibrated products are PDS4 and ISIS-importable (v10+), so supplied labels also unlock independent azimuth/elevation cross-checks.

---

## I. LOLA vs SLDEM2015 comparison

| Criterion | LOLA `LDEM_512` | SLDEM2015 512 ppd |
| --- | --- | --- |
| Vertical accuracy | ~1 m absolute on tracks (orbit overlap); gridded product degrades where interpolated | 3–4 m typical (TC co-registered to LOLA; 90% tiles <5 m RMS) |
| Horizontal resolution | 59 m posting; effective resolution limited by ~5-shot footprint + interpolation between tracks | ~60 m posting; effective 60–120 m (TC stereo), but continuous (no inter-track interpolation) |
| Hillshade suitability (3–6 km tiles) | Good where dense; equatorial gaps risk faceted/interpolated shading artifacts at OHRC/TMC scales | **Better**: gap-filled, crossover-corrected, designed for orthorectification-grade shading |
| Coverage of six regions | Full (2 tiles) | Full (2 tiles, both inside ±60°) |
| Programmatic ingestion | FLOAT_IMG + PDS label/XML; rasterio PDS driver or manual reshape; integer variant needs scale math | JP2+AUX.XML opens directly in rasterio/GDAL; IMG mirror as fallback |
| Provenance | LOLA team / PDS Geosciences, strongest single-instrument geodetic control | Same PDS dataset + peer-reviewed merge (Barker et al. 2016), LOLA-controlled |
| rasterio compatibility | Straightforward (float IMG or JP2) | Straightforward (JP2+AUX preferred) |

---

## J. Recommended DEM source

**SLDEM2015 512 ppd TILES (JP2 + AUX.XML primary, IMG mirror secondary).**

Why: identical ~60 m posting to LOLA 512, but it removes the single biggest scientific risk for these tiles — equatorial LOLA inter-track gaps. Both footprints are small (≤0.13° lon, ≤0.63° lat) near-equatorial scenes where LOLA-only grids interpolate; SLDEM2015 fills those pixels with co-registered Kaguya TC controlled to the same GRAIL/LOLA frame (3–4 m vertical), keeps full PDS provenance in the same dataset, and offers the easiest rasterio path (GIS-ready JP2). LOLA `LDEM_512_FLOAT` remains the named fallback for a pure-single-instrument control run.

---

## K. Exact external inputs still required

1. DEM rasters (operator-supplied, NOT downloaded here): `SLDEM2015_512_30S_00S_315_360` + `SLDEM2015_512_00N_30N_225_270` (JP2+AUX.XML+LBL each; or IMG+LBL mirrors), with source URLs, download timestamps, SHA-256, and version confirmation from the `.LBL`/`.XML`.
2. Chandrayaan-2 PDS4 labels: the four XMLs in §H (plus optional two IIRS XMLs), with SHA-256. These unblock solar elevation for 002–006 and observation-time/spacecraft/footprint metadata.
3. Solar elevation values: six OHRC + six TMC-2 `sun_elevation_deg` read from the labels above. No substitutes accepted.
4. (Optional control) LOLA `ldem_512_45s_00s_270_360_float` + `ldem_512_00n_45n_180_270_float` if a LOLA-only comparison is later approved.

Raw DEMs must NOT be committed to git unless tiny and license/storage-explicit; store paths + checksums + manifests instead.

---

## L. Proposed Phase 14 ingestion contract

No code implemented in Phase 13. Specified interface for Phase 14 approval:

Inputs:

- `dem_source`: enum `SLDEM2015_512_TILE` (primary) | `LDEM_512_FLOAT` (control); `product_id` (exact tile stem, e.g. `SLDEM2015_512_30S_00S_315_360`); `dataset_id` (`LRO-L-LOLA-4-GDR-V1.0`); `source_url`, `download_timestamp_utc`, `original_filename`, `sha256`, `product_version` (from LBL/XML).
- `raster_path` (local, outside git), `bbox` (`west_lon/east_lon/south_lat/north_lat` per region), `src_crs` (as labeled; expect cylindrical 0–360 E), `dst_crs` (Moon EQC `+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m` to match tiling), `elevation_units` (must resolve to meters rel. 1737400; record scale/offset applied, if any).
- `solar_azimuth_deg` + `solar_elevation_deg` per sensor (from PDS4 labels only), `working_gsd_m`, `provenance_record` (product + label IDs + code version), `expected_sha256`.

Processing (to be implemented in Phase 14, not here): verify SHA-256 → open with rasterio (JP2+AUX or FLOAT_IMG+label) → `from_bounds` window crop per region → reproject/resample to region grid (record method, e.g. bilinear/cubic, NO Sobel/stereo synthesis) → emit float32 meters.

Outputs per region:

- `elevation_m` float32 raster + `transform`, `crs`, `bounds`, `width/height`, `nodata`, `resampling_method`, `unit_conversion_applied`.
- `provenance.json`: DEM product/dataset/version, source URL, timestamps, filenames, SHA-256 (source + derived), crop bounds/CRS, resampling, solar source + label IDs, region association, code version.
- Registry entry (append-only, checksummed) linking region → DEM tile → solar labels.
- Rejection rules: wrong tile bounds, non-meter units unresolved, missing solar elevation, missing checksum, or any `dem_512.png`/gradient-synthesized input → hard fail.

---

## M. Provenance/reproducibility requirements

Per DEM file + per PDS4 label store: authoritative product identifier, dataset ID (`LRO-L-LOLA-4-GDR-V1.0` for DEMs), source URL, download timestamp (UTC), product/label version + creation time, original filename, SHA-256 (source and derived crop), crop bounds + CRS (src/dst), resampling method, unit conversion (scale/offset or none), solar metadata source (label ID + tag path), PDS4 label identifier, operator identity, and code version. Keep raw rasters out of git; commit only manifests/registries + the audit report. Cite Barker et al. 2016 for SLDEM2015 and Neumann/Smith (LOLA GDR) for LOLA.

---

## N. Final decision

**BLOCKED_BOTH**

- Products: **identified** (2 SLDEM2015 tiles primary + 2 LOLA tiles control, §E) but raster **files + checksums not yet supplied** — ingestion cannot run on identifiers alone.
- Solar metadata: azimuth present; **elevation missing for regions 002–006** with labels offline; historical 14.06°/48.75° covers region_001 evidence only.
- Next step (Phase 14, upon operator supply): implement the §L contract against the supplied files, starting with checksum + bounds + CRS + units validation and a read-only hillshade preview — still no matcher/model/default changes until that preview is reviewed.

---

## Integrity appendix (this phase)

- `python3 -m pytest -q`: **258 passed, 2 skipped** (42s, warnings only — rasterio georeferencing + sklearn unpickle notes).
- `npm --prefix lunar-frontend run smoke`: **8 passed, 0 failed**.
- `git diff -- ML_model/ai_verifier_model.pkl`: empty (unmodified).
- `git diff --stat -- ML_model/ground_truth_matches.json ML_model/matcher_cfog.py ML_model/ai_verifier.py ML_model/train_ai_verifier.py`: empty (all unmodified).
- Production defaults: `use_goa_preselection=False` (function default, `matcher_cfog.py:1436`); candidate-generation `half_p=8` convention untouched (no eval/production edits in this phase).
- Held-out groups: untouched (no training/eval reruns touching splits).
- Files created: `evaluation/phase13_authoritative_dem_product_audit.md` (this report). Production code: unmodified. `git status` shows only this new untracked report.
