"""
tests/test_lro_pds3_parser.py — Tests for LRO NAC PDS3 Metadata Parsing
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from lro_pds3_parser import parse_pds3_text, read_pds3_label, extract_lro_nac_metadata
from metadata import extract_sensor_metadata, normalize_sensor_name


SAMPLE_PDS3_LBL = """
PDS_VERSION_ID                     = PDS3

/* FILE CHARACTERISTICS */
RECORD_TYPE                        = FIXED_LENGTH
RECORD_BYTES                       = 5064
FILE_RECORDS                       = 52225
LABEL_RECORDS                      = 1
^IMAGE                             = 2

/* DATA IDENTIFICATION */
DATA_SET_ID                        = "LRO-L-LROC-3-CDR-V1.0"
ORIGINAL_PRODUCT_ID                = nacl002b1661
PRODUCT_ID                         = M1417670274LC
MISSION_NAME                       = "LUNAR RECONNAISSANCE ORBITER"
INSTRUMENT_HOST_NAME               = "LUNAR RECONNAISSANCE ORBITER"
INSTRUMENT_HOST_ID                 = LRO
INSTRUMENT_NAME                    = "LUNAR RECONNAISSANCE ORBITER CAMERA"
INSTRUMENT_ID                      = LROC
START_TIME                         = 2022-09-13T01:03:26.869
STOP_TIME                          = 2022-09-13T01:03:58.456
ORBIT_NUMBER                       = 59488
FRAME_ID                           = LEFT

/* VIEWING GEOMETRY */
INCIDENCE_ANGLE                    = 5.82 <deg>
EMISSION_ANGLE                     = 1.71 <deg>
PHASE_ANGLE                        = 7.17 <deg>
SOLAR_AZIMUTH_ANGLE                = 85.4 <deg>

/* MAP PROJECTION */
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE              = "EQUIDISTANT CYLINDRICAL"
  MAP_SCALE                        = 0.914 <m>
  LINE_PROJECTION_OFFSET           = 1024.5
  SAMPLE_PROJECTION_OFFSET         = 512.5
  MINIMUM_LATITUDE                 = -3.81
  MAXIMUM_LATITUDE                 = -2.18
  WESTERNMOST_LONGITUDE            = 336.37
  EASTERNMOST_LONGITUDE            = 336.64
END_OBJECT = IMAGE_MAP_PROJECTION

/* DATA OBJECT */
OBJECT = IMAGE
  LINES                 = 52224
  LINE_SAMPLES          = 5064
  SAMPLE_BITS           = 16
  SAMPLE_TYPE           = LSB_INTEGER
END_OBJECT = IMAGE
END
"""


def test_normalize_sensor_name_lro_nac():
    assert normalize_sensor_name("LRO_NAC") == "LRO_NAC"
    assert normalize_sensor_name("NAC") == "LRO_NAC"
    assert normalize_sensor_name("lro_nac_ref") == "LRO_NAC"


def test_parse_pds3_text():
    data = parse_pds3_text(SAMPLE_PDS3_LBL)
    assert data["PDS_VERSION_ID"] == "PDS3"
    assert data["PRODUCT_ID"] == "M1417670274LC"
    assert data["INSTRUMENT_ID"] == "LROC"
    assert data["INCIDENCE_ANGLE"] == 5.82
    assert data["EMISSION_ANGLE"] == 1.71
    assert data["IMAGE_MAP_PROJECTION"]["MAP_SCALE"] == 0.914
    assert data["IMAGE_MAP_PROJECTION"]["MINIMUM_LATITUDE"] == -3.81
    assert data["IMAGE"]["LINES"] == 52224


def test_extract_lro_nac_metadata_from_lbl(tmp_path):
    lbl_file = tmp_path / "M1417670274LC.LBL"
    lbl_file.write_text(SAMPLE_PDS3_LBL)

    meta = extract_lro_nac_metadata(lbl_file)
    assert meta.sensor == "LRO_NAC"
    assert meta.gsd_m == pytest.approx(0.914)
    assert meta.incidence_angle_deg == pytest.approx(5.82)
    assert meta.emission_angle_deg == pytest.approx(1.71)
    assert meta.sun_azimuth_deg == pytest.approx(85.4)
    assert meta.phase_angle_deg == pytest.approx(7.17)
    assert meta.acquisition_time == "2022-09-13T01:03:26.869"
    assert meta.bounds == (336.37, 336.64, -3.81, -2.18)
    assert meta.provenance["gsd_m"] == "header"
    assert meta.provenance["incidence_angle_deg"] == "header"


def test_extract_sensor_metadata_integration(tmp_path):
    # Test that extract_sensor_metadata automatically finds .LBL sidecar
    img_file = tmp_path / "nac_reference.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # dummy png header
    lbl_file = tmp_path / "nac_reference.lbl"
    lbl_file.write_text(SAMPLE_PDS3_LBL)

    meta = extract_sensor_metadata(img_file)
    assert meta.sensor == "LRO_NAC"
    assert meta.gsd_m == pytest.approx(0.914)
    assert meta.incidence_angle_deg == pytest.approx(5.82)
