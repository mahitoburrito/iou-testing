#!/usr/bin/env python3
"""
Convert Lucid JSON calibration format to KITTI .txt format.

Lucid format contains:
- intrinsic_params: fx, fy, cx, cy, distortion coefficients
- extrinsic_params: quaternion (x, y, z, w), translation (px, py, pz)

KITTI format contains:
- P0: 3x4 intrinsic projection matrix
- Tr_velo_to_cam: 3x4 extrinsic transformation matrix (lidar to camera)

Usage:
    python lucid_to_kitti.py --input lucid_calib.json --output calib.txt
    python lucid_to_kitti.py --input lucid_calib.json --output calib.txt --no-invert
"""

import argparse
import json
import numpy as np
from typing import Dict, Tuple


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    
    Note: Lucid uses (x, y, z, w) order, not (w, x, y, z).
    """
    # Normalize
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    
    return R


def build_intrinsic_matrix(intrinsic: Dict) -> np.ndarray:
    """
    Build 3x4 intrinsic projection matrix from Lucid intrinsic params.
    
    Args:
        intrinsic: Dict with fx, fy, cx, cy
        
    Returns:
        3x4 projection matrix
    """
    fx = intrinsic.get("fx", intrinsic.get("focal_length_x", 0))
    fy = intrinsic.get("fy", intrinsic.get("focal_length_y", 0))
    cx = intrinsic.get("cx", intrinsic.get("principal_point_x", 0))
    cy = intrinsic.get("cy", intrinsic.get("principal_point_y", 0))
    
    P = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0]
    ])
    
    return P


def build_extrinsic_matrix(extrinsic: Dict, invert: bool = True) -> np.ndarray:
    """
    Build 4x4 extrinsic transformation matrix from Lucid extrinsic params.
    
    Args:
        extrinsic: Dict with quaternion (x, y, z, w) and translation (px, py, pz)
        invert: If True, invert the matrix (Lucid OPTICAL format is camera-to-world,
                KITTI expects world-to-camera / lidar-to-camera)
                
    Returns:
        4x4 transformation matrix
    """
    # Extract quaternion
    quat = extrinsic.get("quaternion", {})
    qx = quat.get("x", 0)
    qy = quat.get("y", 0)
    qz = quat.get("z", 0)
    qw = quat.get("w", 1)
    
    # Extract translation
    px = extrinsic.get("px", extrinsic.get("x", 0))
    py = extrinsic.get("py", extrinsic.get("y", 0))
    pz = extrinsic.get("pz", extrinsic.get("z", 0))
    
    # Build rotation matrix
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]
    
    # Invert if needed (Lucid OPTICAL is camera-to-world, KITTI is world-to-camera)
    if invert:
        T = np.linalg.inv(T)
    
    return T


def load_lucid_calib(input_path: str) -> Dict:
    """Load Lucid calibration from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def save_kitti_calib(
    output_path: str, 
    intrinsic: np.ndarray, 
    extrinsic: np.ndarray
):
    """
    Save calibration in KITTI format.
    
    Args:
        output_path: Output .txt file path
        intrinsic: 3x4 intrinsic projection matrix
        extrinsic: 4x4 extrinsic transformation matrix
    """
    intrinsic_flat = " ".join(f"{v}" for v in intrinsic.flatten())
    extrinsic_flat = " ".join(f"{v}" for v in extrinsic[:3, :].flatten())
    
    with open(output_path, "w") as f:
        f.write(f"P0: {intrinsic_flat}\n")
        f.write(f"P1: {intrinsic_flat}\n")
        f.write(f"P2: {intrinsic_flat}\n")
        f.write(f"P3: {intrinsic_flat}\n")
        f.write("R0_rect: 0 0 0 0 0 0 0 0 0\n")
        f.write(f"Tr_velo_to_cam: {extrinsic_flat}\n")
        f.write("Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0\n")


def convert_lucid_to_kitti(
    input_path: str, 
    output_path: str, 
    invert: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Lucid JSON calibration to KITTI .txt format.
    
    Args:
        input_path: Path to Lucid JSON calibration file
        output_path: Path to output KITTI .txt file
        invert: Whether to invert the extrinsic matrix
        verbose: Print conversion details
        
    Returns:
        Tuple of (intrinsic, extrinsic) matrices
    """
    # Load Lucid calibration
    lucid_calib = load_lucid_calib(input_path)
    
    # Extract intrinsic and extrinsic params
    intrinsic_params = lucid_calib.get("intrinsic_params", lucid_calib.get("intrinsic", {}))
    extrinsic_params = lucid_calib.get("extrinsic_params", lucid_calib.get("extrinsic", {}))
    
    # Build matrices
    intrinsic = build_intrinsic_matrix(intrinsic_params)
    extrinsic = build_extrinsic_matrix(extrinsic_params, invert=invert)
    
    # Save to KITTI format
    save_kitti_calib(output_path, intrinsic, extrinsic)
    
    if verbose:
        camera = intrinsic_params.get("camera", "unknown")
        coord_system = extrinsic_params.get("camera_coordinate", "unknown")
        
        print(f"Converted: {input_path} -> {output_path}")
        print(f"  Camera: {camera}")
        print(f"  Coordinate system: {coord_system}")
        print(f"  Extrinsic inverted: {invert}")
        print(f"\n  Intrinsic (P0):")
        print(f"    fx={intrinsic[0,0]:.2f}, fy={intrinsic[1,1]:.2f}")
        print(f"    cx={intrinsic[0,2]:.2f}, cy={intrinsic[1,2]:.2f}")
        print(f"\n  Extrinsic (4x4):")
        for row in extrinsic:
            print(f"    [{', '.join(f'{v:10.6f}' for v in row)}]")
    
    return intrinsic, extrinsic


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lucid JSON calibration to KITTI .txt format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input Lucid JSON calibration file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output KITTI .txt calibration file"
    )
    parser.add_argument(
        "--no-invert", action="store_true",
        help="Don't invert the extrinsic matrix (use if already in lidar-to-camera format)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    convert_lucid_to_kitti(
        input_path=args.input,
        output_path=args.output,
        invert=not args.no_invert,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
