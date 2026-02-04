#!/usr/bin/env python3
"""
Convert calibration files between different formats.

Supports conversions between:
- KITTI format (per-frame .txt files)
- Lucid production format (with offset YAML files)
- Single consolidated extrinsic file

Also supports:
- Applying quaternion-based calibration adjustments
- Extracting differences between calibrations
- Converting directory structures

Usage:
    # Convert KITTI format to Lucid format with offsets
    python convert_calib.py kitti_to_lucid --input ./data/processed/calib --output ./lucid_calib
    
    # Apply calibration adjustment to existing calibration
    python convert_calib.py apply_offset --input ./calib.txt --offset ./offset.yaml --output ./adjusted_calib.txt
    
    # Compare two calibrations and extract the difference as offset
    python convert_calib.py extract_offset --original ./original_calib.txt --adjusted ./new_calib.txt --output ./offset.yaml
    
    # Convert to single consolidated file per camera
    python convert_calib.py consolidate --input ./data/processed/calib --output ./consolidated
"""

import argparse
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import yaml


# Camera name to folder name mapping
CAMERA_MAP = {
    "FNC": "cam-03",
    "FWC_C": "cam-02",
    "FWC_L": "cam-07",
    "FWC_R": "cam-05",
    "RNC_C": "cam-06",
    "RNC_L": "cam-08",
    "RNC_R": "cam-04",
}

FOLDER_TO_CAMERA = {v: k for k, v in CAMERA_MAP.items()}


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to quaternion (w, x, y, z).
    
    Uses the Shepperd method for numerical stability.
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    # Normalize
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    return (w/norm, x/norm, y/norm, z/norm)


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    
    Matches the implementation in utils.py.
    """
    # Normalize
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    
    return R


def load_kitti_calib(calib_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load calibration from KITTI-format .txt file.
    
    Returns:
        intrinsic: 3x4 projection matrix
        extrinsic: 4x4 transformation matrix
    """
    intrinsic = None
    extrinsic = None
    
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            key = parts[0].rstrip(":")
            values = [float(v) for v in parts[1:] if v]
            
            if key == "P0":
                intrinsic = np.array(values).reshape(3, 4)
            elif key == "Tr_velo_to_cam":
                extrinsic = np.array(values + [0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
    
    return intrinsic, extrinsic


def save_kitti_calib(
    calib_file: str,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
):
    """
    Save calibration in KITTI format.
    
    Args:
        calib_file: Output file path
        intrinsic: 3x4 projection matrix
        extrinsic: 4x4 transformation matrix
    """
    # Flatten matrices
    intrinsic_flat = " ".join(f"{v}" for v in intrinsic.flatten())
    extrinsic_flat = " ".join(f"{v}" for v in extrinsic[:3, :].flatten())
    
    with open(calib_file, "w") as f:
        f.write(f"P0: {intrinsic_flat}\n")
        f.write(f"P1: {intrinsic_flat}\n")
        f.write(f"P2: {intrinsic_flat}\n")
        f.write(f"P3: {intrinsic_flat}\n")
        f.write("R0_rect: 0 0 0 0 0 0 0 0 0\n")
        f.write(f"Tr_velo_to_cam: {extrinsic_flat}\n")
        f.write("Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0\n")


def load_offset_yaml(offset_file: str, camera: str) -> Dict:
    """
    Load offset parameters from Lucid-format YAML file.
    
    Expected format:
        FWC_C:
            dx: 0.0
            dy: 0.0
            dz: 0.0
            qw: 1.0
            qx: 0.0
            qy: 0.0
            qz: 0.0
    """
    with open(offset_file, "r") as f:
        data = yaml.safe_load(f)
    
    if camera in data:
        return data[camera]
    
    return {"dx": 0, "dy": 0, "dz": 0, "qw": 1, "qx": 0, "qy": 0, "qz": 0}


def save_offset_yaml(
    offset_file: str,
    offsets: Dict[str, Dict],
):
    """
    Save offset parameters to Lucid-format YAML file.
    
    Args:
        offset_file: Output file path
        offsets: Dict mapping camera name to offset dict
    """
    with open(offset_file, "w") as f:
        yaml.dump(offsets, f, default_flow_style=False)


def apply_calibration_adjustment(
    calib_matrix: np.ndarray,
    dx: float, dy: float, dz: float,
    qw: float, qx: float, qy: float, qz: float,
) -> np.ndarray:
    """
    Apply translation and rotation adjustment to a 4x4 calibration matrix.
    
    This matches the implementation in utils.py.
    
    Args:
        calib_matrix: 4x4 homogeneous transformation matrix
        dx, dy, dz: Translation adjustments
        qw, qx, qy, qz: Quaternion rotation adjustment
        
    Returns:
        Adjusted 4x4 calibration matrix
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    
    # Create adjustment matrix
    adjustment = np.eye(4)
    adjustment[:3, :3] = R
    adjustment[:3, 3] = [dx, dy, dz]
    
    # Apply adjustment (matches utils.py convention)
    adjusted = np.linalg.inv(adjustment) @ calib_matrix
    
    return adjusted


def extract_offset_from_calibrations(
    original: np.ndarray,
    adjusted: np.ndarray,
) -> Dict:
    """
    Extract the offset parameters that transform original to adjusted.
    
    Args:
        original: Original 4x4 extrinsic matrix
        adjusted: Adjusted 4x4 extrinsic matrix
        
    Returns:
        Dict with dx, dy, dz, qw, qx, qy, qz
    """
    # Compute the adjustment matrix
    # From: adjusted = inv(adjustment) @ original
    # So: adjustment = original @ inv(adjusted)
    adjustment = original @ np.linalg.inv(adjusted)
    
    # Extract translation
    dx, dy, dz = adjustment[:3, 3]
    
    # Extract rotation and convert to quaternion
    R = adjustment[:3, :3]
    qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
    
    return {
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "qw": float(qw),
        "qx": float(qx),
        "qy": float(qy),
        "qz": float(qz),
    }


def compare_calibrations(
    calib1: np.ndarray,
    calib2: np.ndarray,
) -> Dict:
    """
    Compare two calibration matrices and return difference metrics.
    
    Returns:
        Dict with translation_diff, rotation_diff (in degrees), etc.
    """
    # Translation difference
    t1 = calib1[:3, 3]
    t2 = calib2[:3, 3]
    translation_diff = np.linalg.norm(t2 - t1)
    
    # Rotation difference
    R1 = calib1[:3, :3]
    R2 = calib2[:3, :3]
    R_diff = R2 @ R1.T
    
    # Convert to angle (axis-angle representation)
    trace = np.trace(R_diff)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle_rad)
    
    return {
        "translation_diff_m": float(translation_diff),
        "rotation_diff_deg": float(angle_deg),
        "translation_x_diff": float(t2[0] - t1[0]),
        "translation_y_diff": float(t2[1] - t1[1]),
        "translation_z_diff": float(t2[2] - t1[2]),
    }


def find_camera_from_folder(folder_name: str) -> Optional[str]:
    """Extract camera name from folder name."""
    for cam_id, cam_name in FOLDER_TO_CAMERA.items():
        if cam_id in folder_name:
            return cam_name
    return None


# =============================================================================
# Command: kitti_to_lucid
# =============================================================================

def cmd_kitti_to_lucid(args):
    """Convert KITTI-format calibration to Lucid format with offset files."""
    input_dir = args.input
    output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all sessions
    sessions = [d for d in os.listdir(input_dir) 
                if osp.isdir(osp.join(input_dir, d)) and not d.startswith('.')]
    
    for session in sessions:
        session_input = osp.join(input_dir, session)
        session_output = osp.join(output_dir, session)
        os.makedirs(session_output, exist_ok=True)
        
        # Create offsets directory
        offsets_dir = osp.join(session_output, "offsets")
        os.makedirs(offsets_dir, exist_ok=True)
        
        # Find camera folders
        cam_folders = [d for d in os.listdir(session_input)
                       if osp.isdir(osp.join(session_input, d)) and "cam" in d.lower()]
        
        for cam_folder in cam_folders:
            camera = find_camera_from_folder(cam_folder)
            if not camera:
                continue
            
            cam_input_dir = osp.join(session_input, cam_folder)
            cam_output_dir = osp.join(session_output, cam_folder)
            os.makedirs(cam_output_dir, exist_ok=True)
            
            # Get first calibration file as reference
            calib_files = sorted([f for f in os.listdir(cam_input_dir) if f.endswith('.txt')])
            if not calib_files:
                continue
            
            # Copy calibration files
            for calib_file in calib_files:
                src = osp.join(cam_input_dir, calib_file)
                dst = osp.join(cam_output_dir, calib_file)
                
                intrinsic, extrinsic = load_kitti_calib(src)
                if intrinsic is not None and extrinsic is not None:
                    save_kitti_calib(dst, intrinsic, extrinsic)
            
            # Create identity offset file for this camera
            offset_file = osp.join(offsets_dir, f"{cam_folder}.yaml")
            offsets = {
                camera: {
                    "dx": 0.0, "dy": 0.0, "dz": 0.0,
                    "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                }
            }
            save_offset_yaml(offset_file, offsets)
            
            print(f"Converted: {session}/{cam_folder}")
    
    print(f"\nOutput written to: {output_dir}")


# =============================================================================
# Command: apply_offset
# =============================================================================

def cmd_apply_offset(args):
    """Apply calibration offset to an extrinsic file."""
    # Load original calibration
    intrinsic, extrinsic = load_kitti_calib(args.input)
    
    if extrinsic is None:
        print(f"Error: Could not load extrinsic from {args.input}")
        return
    
    # Load or parse offset
    if args.offset.endswith('.yaml') or args.offset.endswith('.yml'):
        offset_data = load_offset_yaml(args.offset, args.camera or "default")
    else:
        # Parse as JSON or comma-separated values
        try:
            offset_data = json.loads(args.offset)
        except json.JSONDecodeError:
            print("Error: Offset must be a YAML file or JSON string")
            return
    
    # Apply offset
    adjusted = apply_calibration_adjustment(
        extrinsic,
        offset_data.get("dx", 0),
        offset_data.get("dy", 0),
        offset_data.get("dz", 0),
        offset_data.get("qw", 1),
        offset_data.get("qx", 0),
        offset_data.get("qy", 0),
        offset_data.get("qz", 0),
    )
    
    # Save result
    save_kitti_calib(args.output, intrinsic, adjusted)
    print(f"Adjusted calibration saved to: {args.output}")
    
    # Print comparison
    diff = compare_calibrations(extrinsic, adjusted)
    print(f"\nAdjustment applied:")
    print(f"  Translation change: {diff['translation_diff_m']:.6f} m")
    print(f"  Rotation change: {diff['rotation_diff_deg']:.6f} degrees")


# =============================================================================
# Command: extract_offset
# =============================================================================

def cmd_extract_offset(args):
    """Extract offset parameters from two calibration files."""
    # Load both calibrations
    _, original_ext = load_kitti_calib(args.original)
    _, adjusted_ext = load_kitti_calib(args.adjusted)
    
    if original_ext is None or adjusted_ext is None:
        print("Error: Could not load calibration files")
        return
    
    # Extract offset
    offset = extract_offset_from_calibrations(original_ext, adjusted_ext)
    
    # Print comparison
    diff = compare_calibrations(original_ext, adjusted_ext)
    print(f"Calibration difference:")
    print(f"  Translation: {diff['translation_diff_m']:.6f} m")
    print(f"    dx: {diff['translation_x_diff']:.6f}")
    print(f"    dy: {diff['translation_y_diff']:.6f}")
    print(f"    dz: {diff['translation_z_diff']:.6f}")
    print(f"  Rotation: {diff['rotation_diff_deg']:.6f} degrees")
    
    # Save offset
    if args.output.endswith('.yaml') or args.output.endswith('.yml'):
        camera = args.camera or "default"
        save_offset_yaml(args.output, {camera: offset})
    else:
        with open(args.output, 'w') as f:
            json.dump(offset, f, indent=2)
    
    print(f"\nOffset saved to: {args.output}")


# =============================================================================
# Command: consolidate
# =============================================================================

def cmd_consolidate(args):
    """Consolidate per-frame calibrations to single file per camera."""
    input_dir = args.input
    output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all sessions
    sessions = [d for d in os.listdir(input_dir) 
                if osp.isdir(osp.join(input_dir, d)) and not d.startswith('.')]
    
    for session in sessions:
        session_input = osp.join(input_dir, session)
        session_output = osp.join(output_dir, session)
        os.makedirs(session_output, exist_ok=True)
        
        # Find camera folders
        cam_folders = [d for d in os.listdir(session_input)
                       if osp.isdir(osp.join(session_input, d)) and "cam" in d.lower()]
        
        session_data = {}
        
        for cam_folder in cam_folders:
            camera = find_camera_from_folder(cam_folder)
            if not camera:
                continue
            
            cam_input_dir = osp.join(session_input, cam_folder)
            
            # Get all calibration files
            calib_files = sorted([f for f in os.listdir(cam_input_dir) if f.endswith('.txt')])
            if not calib_files:
                continue
            
            # Use first file as reference (or compute average)
            if args.method == "first":
                ref_file = osp.join(cam_input_dir, calib_files[0])
                intrinsic, extrinsic = load_kitti_calib(ref_file)
            elif args.method == "average":
                # Average all calibrations
                intrinsics = []
                extrinsics = []
                for cf in calib_files:
                    intr, extr = load_kitti_calib(osp.join(cam_input_dir, cf))
                    if intr is not None and extr is not None:
                        intrinsics.append(intr)
                        extrinsics.append(extr)
                intrinsic = np.mean(intrinsics, axis=0)
                extrinsic = np.mean(extrinsics, axis=0)
            
            # Save consolidated calibration
            output_file = osp.join(session_output, f"{camera}_calib.txt")
            save_kitti_calib(output_file, intrinsic, extrinsic)
            
            session_data[camera] = {
                "intrinsic": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
                "source_files": len(calib_files),
            }
            
            print(f"Consolidated: {session}/{camera} ({len(calib_files)} files)")
        
        # Also save as JSON
        json_file = osp.join(session_output, "calibration.json")
        with open(json_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    print(f"\nOutput written to: {output_dir}")


# =============================================================================
# Command: compare
# =============================================================================

def cmd_compare(args):
    """Compare two calibration files and show differences."""
    _, ext1 = load_kitti_calib(args.file1)
    _, ext2 = load_kitti_calib(args.file2)
    
    if ext1 is None or ext2 is None:
        print("Error: Could not load calibration files")
        return
    
    diff = compare_calibrations(ext1, ext2)
    
    print(f"Comparison: {args.file1} vs {args.file2}")
    print(f"\nDifferences:")
    print(f"  Total translation: {diff['translation_diff_m']:.6f} m")
    print(f"    X: {diff['translation_x_diff']:.6f} m")
    print(f"    Y: {diff['translation_y_diff']:.6f} m")
    print(f"    Z: {diff['translation_z_diff']:.6f} m")
    print(f"  Total rotation: {diff['rotation_diff_deg']:.6f} degrees")
    
    # Extract and show offset
    offset = extract_offset_from_calibrations(ext1, ext2)
    print(f"\nOffset to transform file1 -> file2:")
    print(f"  dx: {offset['dx']:.6f}")
    print(f"  dy: {offset['dy']:.6f}")
    print(f"  dz: {offset['dz']:.6f}")
    print(f"  qw: {offset['qw']:.6f}")
    print(f"  qx: {offset['qx']:.6f}")
    print(f"  qy: {offset['qy']:.6f}")
    print(f"  qz: {offset['qz']:.6f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert calibration files between formats"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # kitti_to_lucid command
    p_ktl = subparsers.add_parser(
        "kitti_to_lucid",
        help="Convert KITTI format to Lucid format with offset files"
    )
    p_ktl.add_argument("--input", "-i", required=True, help="Input calibration directory")
    p_ktl.add_argument("--output", "-o", required=True, help="Output directory")
    
    # apply_offset command
    p_apply = subparsers.add_parser(
        "apply_offset",
        help="Apply calibration offset to an extrinsic file"
    )
    p_apply.add_argument("--input", "-i", required=True, help="Input calibration file")
    p_apply.add_argument("--offset", required=True, help="Offset file (YAML) or JSON string")
    p_apply.add_argument("--output", "-o", required=True, help="Output calibration file")
    p_apply.add_argument("--camera", help="Camera name (for multi-camera offset files)")
    
    # extract_offset command
    p_extract = subparsers.add_parser(
        "extract_offset",
        help="Extract offset parameters from two calibration files"
    )
    p_extract.add_argument("--original", required=True, help="Original calibration file")
    p_extract.add_argument("--adjusted", required=True, help="Adjusted calibration file")
    p_extract.add_argument("--output", "-o", required=True, help="Output offset file")
    p_extract.add_argument("--camera", help="Camera name for YAML output")
    
    # consolidate command
    p_cons = subparsers.add_parser(
        "consolidate",
        help="Consolidate per-frame calibrations to single file per camera"
    )
    p_cons.add_argument("--input", "-i", required=True, help="Input calibration directory")
    p_cons.add_argument("--output", "-o", required=True, help="Output directory")
    p_cons.add_argument(
        "--method", choices=["first", "average"], default="first",
        help="Method to consolidate (first frame or average)"
    )
    
    # compare command
    p_cmp = subparsers.add_parser(
        "compare",
        help="Compare two calibration files"
    )
    p_cmp.add_argument("file1", help="First calibration file")
    p_cmp.add_argument("file2", help="Second calibration file")
    
    args = parser.parse_args()
    
    if args.command == "kitti_to_lucid":
        cmd_kitti_to_lucid(args)
    elif args.command == "apply_offset":
        cmd_apply_offset(args)
    elif args.command == "extract_offset":
        cmd_extract_offset(args)
    elif args.command == "consolidate":
        cmd_consolidate(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
