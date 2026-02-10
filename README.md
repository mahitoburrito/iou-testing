# IoU Testing - Calibration Quality Evaluation

Tools for evaluating camera-lidar calibration quality by computing IoU (Intersection over Union) between 2D bounding box annotations and projected 3D bounding boxes.

## Overview

This project provides utilities to:
- **Quantify calibration quality** by measuring how well 3D bounding boxes project onto 2D image annotations
- **Compare calibration methods** by running IoU metrics on original vs. fine-tuned calibrations
- **Convert calibration formats** between KITTI format and Lucid's production format

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run IoU computation on all data
python compute_iou.py --data_dir ./data --output results.csv

# Run on a single camera, single frame
python compute_iou.py --data_dir ./data --cameras FNC --frames 1 --detailed
```

---

## Scripts

### `compute_iou.py` - Main IoU Computation

Calculates IoU metrics between 2D annotations and projected 3D bounding boxes using camera calibration.

**Usage:**

```bash
# Process all sessions, cameras, and frames
python compute_iou.py --data_dir ./data --output results.csv

# Compare with a different calibration
python compute_iou.py --data_dir ./data --calib_dir ./new_calibration --output comparison.csv

# Process specific session
python compute_iou.py --data_dir ./data --sessions xIYH05-gravity-USA-vin116-20250828_221259

# Process specific camera(s)
python compute_iou.py --data_dir ./data --cameras FWC_C FNC

# Process specific frame(s)
python compute_iou.py --data_dir ./data --frames 1 10 50

# Single session, single camera, single frame with detailed output
python compute_iou.py --data_dir ./data \
    --sessions xIYH05-gravity-USA-vin116-20250828_221259 \
    --cameras FNC \
    --frames 1 \
    --detailed \
    --output single_test.csv
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--data_dir` | Root data directory (default: `./data`) |
| `--calib_dir` | Override calibration directory (for comparing calibrations) |
| `--output` | Output file path (.csv or .json) |
| `--sessions` | Specific session(s) to process |
| `--cameras` | Specific camera(s): FNC, FWC_C, FWC_L, FWC_R, RNC_C, RNC_L, RNC_R |
| `--frames` | Specific frame ID(s) to process |
| `--filter_types` | Object types to include (e.g., `vehicle.small_vehicle.sedan`) |
| `--detailed` | Include per-object results in output |
| `--quiet` | Suppress progress output |

**Output:**

CSV with columns: `session, camera, mean_iou, median_iou, std_iou, min_iou, max_iou, count`

With `--detailed`, also includes per-object results.

---

### `convert_calib.py` - Calibration Format Converter

Converts between calibration formats and applies/extracts calibration adjustments.

**Commands:**

```bash
# Compare two calibration files
python convert_calib.py compare ./original_calib.txt ./new_calib.txt

# Extract offset between two calibrations (outputs quaternion + translation)
python convert_calib.py extract_offset \
    --original ./original.txt \
    --adjusted ./fine_tuned.txt \
    --output offset.yaml

# Apply an offset to a calibration
python convert_calib.py apply_offset \
    --input ./calib.txt \
    --offset ./offset.yaml \
    --output ./adjusted.txt

# Convert KITTI format to Lucid format with offset files
python convert_calib.py kitti_to_lucid \
    --input ./data/processed/calib \
    --output ./lucid_format_calib

# Consolidate per-frame calibrations to single file per camera
python convert_calib.py consolidate \
    --input ./data/processed/calib \
    --output ./consolidated \
    --method first  # or 'average'
```

---

### `lucid_to_kitti.py` - Lucid JSON to KITTI Format Converter

Converts Lucid's JSON calibration format to KITTI .txt format for use with `compute_iou.py`.

**Usage:**

```bash
# Basic conversion (automatically inverts extrinsic for OPTICAL coordinate system)
python lucid_to_kitti.py --input lucid_calib.json --output calib.txt

# Don't invert extrinsic (if already in lidar-to-camera format)
python lucid_to_kitti.py --input lucid_calib.json --output calib.txt --no-invert

# Quiet mode
python lucid_to_kitti.py -i lucid_calib.json -o calib.txt -q
```

**Input format (Lucid JSON):**
```json
{
  "intrinsic_params": {
    "fx": 4567.32, "fy": 4566.75, "cx": 1915.45, "cy": 1103.16,
    "camera": "fnc_c"
  },
  "extrinsic_params": {
    "px": 2.457, "py": -0.008, "pz": -0.525,
    "quaternion": { "x": -0.497, "y": 0.500, "z": -0.502, "w": 0.501 },
    "camera_coordinate": "OPTICAL"
  }
}
```

**Output format (KITTI .txt):**
```
P0: fx 0 cx 0 0 fy cy 0 0 0 1 0
Tr_velo_to_cam: r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
```

**Note:** Lucid's `"camera_coordinate": "OPTICAL"` means the extrinsic is camera-to-world. The script automatically inverts it to get lidar-to-camera for KITTI format.

---

## Existing Lucid Production Scripts

These scripts are from Lucid's production environment for reference:

### `sanity_check.py`

Production pipeline for validating 2D/3D label quality and consistency. Runs as part of an Airflow pipeline.

**Features:**
- Downloads batch data from S3
- Validates truncation/occlusion attributes
- Matches 2D labels to 3D labels
- Computes IoU between 2D boxes and projected 3D boxes
- Flags errors and generates debug images
- Uploads results back to S3

### `utils.py`

Core utilities for sensor fusion data processing:
- `quaternion_to_rotation_matrix()` - Convert quaternion to rotation matrix
- `apply_calibration_adjustment()` - Apply translation/rotation offset to calibration
- `get_calib()` - Load KITTI-format calibration files
- `get_bbox_corners()` - Compute 3D bounding box corners
- `calculate_iou()` - Calculate IoU between 2D box and projected 3D hull
- `load_data()` - Load batch data (images, labels, calibration)
- Camera mapping between names (FNC, FWC_C, etc.) and folder names (cam-03, cam-02, etc.)

### `common.py`

Data models and enums:
- `Vendor` - Annotation vendor enum (CODA, SCALE, AVALA)
- `Project` - Project enum (SF_BEV = Sensor Fusion Bird's Eye View)
- `S3Path` - Dataclass for S3 URIs with utilities
- `HpcPath` - Dataclass for mapping S3 paths to HPC paths

### `common_utils.py`

Configuration and AWS utilities:
- `get_config()` - Load YAML configuration files
- `get_aws_session()` - Create boto3 AWS sessions

---

## Data Structure

```
data/
├── processed/
│   ├── KITTI_SENSORFUSION/          # 3D labels (one .txt per session)
│   │   └── <session>.txt
│   ├── KITTI_CAM_FRAME/             # 2D labels (one .txt per camera per frame)
│   │   └── <session>/
│   │       └── <camera_folder>/
│   │           └── <camera>-<frame>.txt
│   └── calib/                       # Calibration (one .txt per camera per frame)
│       └── <session>/
│           └── <camera_folder>/
│               └── <camera>-<frame>.txt
├── processed_2d/                    # Parquet files (alternative format)
│   └── <session>/
│       └── <camera>-<timestamp>.parquet
└── <camera_folder>/                 # Raw images
    └── png_files/
        └── <camera>-<frame>.png
```

### Calibration File Format (KITTI)

```
P0: <12 floats - 3x4 intrinsic projection matrix>
P1: <12 floats>
P2: <12 floats>
P3: <12 floats>
R0_rect: <9 floats - rectification matrix>
Tr_velo_to_cam: <12 floats - 3x4 extrinsic lidar-to-camera transform>
Tr_imu_to_velo: <12 floats>
```

### Camera Mapping

| Camera Name | Folder ID | Description |
|-------------|-----------|-------------|
| FNC | cam-03 | Front Narrow Camera |
| FWC_C | cam-02 | Front Wide Camera Center |
| FWC_L | cam-07 | Front Wide Camera Left |
| FWC_R | cam-05 | Front Wide Camera Right |
| RNC_C | cam-06 | Rear Narrow Camera Center |
| RNC_L | cam-08 | Rear Narrow Camera Left |
| RNC_R | cam-04 | Rear Narrow Camera Right |

---

## Workflow: Comparing Calibrations

1. **Run IoU with original calibration:**
   ```bash
   python compute_iou.py --data_dir ./data --output baseline_iou.csv
   ```

2. **Place fine-tuned calibration** in a new directory with the same structure as `./data/processed/calib`

3. **Run IoU with new calibration:**
   ```bash
   python compute_iou.py --data_dir ./data --calib_dir ./fine_tuned_calib --output finetuned_iou.csv
   ```

4. **Compare results** - higher mean IoU indicates better calibration alignment

5. **Extract the calibration difference:**
   ```bash
   python convert_calib.py compare ./data/processed/calib/.../original.txt ./fine_tuned_calib/.../new.txt
   ```

---

## Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- shapely >= 2.0.0
- PyYAML >= 6.0

Install with: `pip install -r requirements.txt`
