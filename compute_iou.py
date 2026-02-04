#!/usr/bin/env python3
"""
Compute IoU metrics for calibration quality evaluation.

This script calculates IoU (Intersection over Union) between 2D bounding box annotations
and projected 3D bounding boxes using provided camera calibration (extrinsic + intrinsic).

Use this to quantitatively compare different calibration methods.

Usage:
    # Process all sessions, cameras, and frames
    python compute_iou.py --data_dir ./data --output results.csv
    
    # Process with a different calibration directory
    python compute_iou.py --data_dir ./data --calib_dir ./new_calibration --output comparison.csv
    
    # Process only one session
    python compute_iou.py --data_dir ./data --sessions xIYH05-gravity-USA-vin116-20250828_221259
    
    # Process only one camera
    python compute_iou.py --data_dir ./data --cameras FWC_C
    
    # Process only specific frames
    python compute_iou.py --data_dir ./data --frames 1 10 50
    
    # Process one session, one camera, one frame
    python compute_iou.py --data_dir ./data --sessions xIYH05-gravity-USA-vin116-20250828_221259 --cameras FNC --frames 1
"""

import argparse
import csv
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


# Camera name to folder name mapping (matching Lucid's convention)
CAMERA_MAP = {
    "FNC": "cam-03",
    "FWC_C": "cam-02",
    "FWC_L": "cam-07",
    "FWC_R": "cam-05",
    "RNC_C": "cam-06",
    "RNC_L": "cam-08",
    "RNC_R": "cam-04",
}

# Reverse mapping
FOLDER_TO_CAMERA = {v: k for k, v in CAMERA_MAP.items()}

# Default image shapes per camera
IMAGE_SHAPES = {
    "FNC": (2160, 3840),
    "FWC_C": (2856, 7680),
    "FWC_L": (2472, 3840),
    "FWC_R": (2472, 3840),
    "RNC_C": (1800, 3200),
    "RNC_L": (1800, 3200),
    "RNC_R": (1800, 3200),
}


@dataclass
class CalibrationData:
    """Camera calibration data."""
    intrinsic: np.ndarray  # 3x4 projection matrix
    extrinsic: np.ndarray  # 4x4 transformation matrix (lidar to camera)
    
    
@dataclass
class BBox3D:
    """3D bounding box in lidar coordinates."""
    track_id: int
    object_type: str
    length: float
    width: float
    height: float
    x: float
    y: float
    z: float
    rotation_z: float
    frame_id: int
    truncated: int = 10
    occluded: int = 10


@dataclass
class BBox2D:
    """2D bounding box in image coordinates."""
    object_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    truncated: int = 10
    occluded: int = 10
    # 3D info stored in 2D label for matching
    height: float = 0
    width: float = 0
    length: float = 0
    loc_x: float = 0
    loc_y: float = 0
    loc_z: float = 0


@dataclass
class IoUResult:
    """IoU computation result for a single object."""
    frame_id: int
    camera: str
    object_type: str
    track_id: int
    iou: float
    distance: float  # distance from ego vehicle
    bbox_2d: Tuple[float, float, float, float]
    projected_valid: bool  # whether 3D projection was valid


@dataclass 
class IoUMetrics:
    """Aggregated IoU metrics."""
    mean_iou: float
    median_iou: float
    std_iou: float
    min_iou: float
    max_iou: float
    count: int
    results: List[IoUResult] = field(default_factory=list)


def load_calibration(calib_file: str) -> CalibrationData:
    """
    Load calibration data from KITTI-format .txt file.
    
    Expected format:
        P0: <12 floats for 3x4 projection matrix>
        Tr_velo_to_cam: <12 floats for 3x4 transformation matrix>
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
                # Convert 3x4 to 4x4 by adding [0, 0, 0, 1] row
                extrinsic = np.array(values + [0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
    
    if intrinsic is None or extrinsic is None:
        raise ValueError(f"Failed to load calibration from {calib_file}")
    
    return CalibrationData(intrinsic=intrinsic, extrinsic=extrinsic)


def load_3d_labels(label_file: str) -> Dict[int, List[BBox3D]]:
    """
    Load 3D labels from KITTI SENSORFUSION format.
    
    Returns dict mapping frame_id -> list of BBox3D objects.
    
    Format per line:
        timestamp frame_id track_id object_type truncated occluded heading 
        length width height location_x location_y location_z rotation_z ...
    """
    labels_by_frame: Dict[int, List[BBox3D]] = {}
    
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            
            frame_id = int(parts[1])
            
            bbox = BBox3D(
                frame_id=frame_id,
                track_id=int(parts[2]),
                object_type=parts[3],
                truncated=int(parts[4]) if parts[4].isdigit() else 10,
                occluded=int(parts[5]) if parts[5].isdigit() else 10,
                length=float(parts[7]),
                width=float(parts[8]),
                height=float(parts[9]),
                x=float(parts[10]),
                y=float(parts[11]),
                z=float(parts[12]),
                rotation_z=float(parts[13]),
            )
            
            if frame_id not in labels_by_frame:
                labels_by_frame[frame_id] = []
            labels_by_frame[frame_id].append(bbox)
    
    return labels_by_frame


def load_2d_labels(label_file: str) -> List[BBox2D]:
    """
    Load 2D labels from KITTI CAM_FRAME format.
    
    Format per line:
        object_type truncated occluded heading x1 y1 x2 y2 
        height width length location_x location_y location_z ...
    """
    labels = []
    
    if not osp.exists(label_file):
        return labels
        
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            
            bbox = BBox2D(
                object_type=parts[0],
                truncated=int(parts[1]) if parts[1].isdigit() else 10,
                occluded=int(parts[2]) if parts[2].isdigit() else 10,
                x1=float(parts[4]),
                y1=float(parts[5]),
                x2=float(parts[6]),
                y2=float(parts[7]),
                height=float(parts[8]),
                width=float(parts[9]),
                length=float(parts[10]),
                loc_x=float(parts[11]),
                loc_y=float(parts[12]),
                loc_z=float(parts[13]),
            )
            labels.append(bbox)
    
    return labels


def get_bbox_corners(bbox: BBox3D) -> np.ndarray:
    """
    Compute 8 corners + center of a 3D bounding box.
    
    Returns: 4x9 array (homogeneous coordinates) for 8 corners + center.
    """
    center = np.array([bbox.x, bbox.y, bbox.z, 1.0])
    l, w, h = bbox.length, bbox.width, bbox.height
    heading = bbox.rotation_z
    
    corners = np.ones((4, 9))
    corners[0, :] = center[0]
    corners[1, :] = center[1]
    corners[2, :] = center[2]
    
    # Apply rotation for length dimension
    corners[0, 0:4] += l/2 * np.cos(heading)
    corners[0, 4:8] -= l/2 * np.cos(heading)
    corners[1, 0:4] += l/2 * np.sin(heading)
    corners[1, 4:8] -= l/2 * np.sin(heading)
    
    # Apply rotation for width dimension
    corners[1, 0:2] += w/2 * np.cos(heading)
    corners[1, 2:4] -= w/2 * np.cos(heading)
    corners[1, 4:6] += w/2 * np.cos(heading)
    corners[1, 6:8] -= w/2 * np.cos(heading)
    
    corners[0, 0:2] -= w/2 * np.sin(heading)
    corners[0, 2:4] += w/2 * np.sin(heading)
    corners[0, 4:6] -= w/2 * np.sin(heading)
    corners[0, 6:8] += w/2 * np.sin(heading)
    
    # Height dimension
    corners[2, 0:8:2] += h/2
    corners[2, 1:8:2] -= h/2
    
    return corners


def project_3d_to_2d(
    corners: np.ndarray, 
    intrinsic: np.ndarray, 
    extrinsic: np.ndarray
) -> Tuple[np.ndarray, List[bool]]:
    """
    Project 3D corners to 2D image coordinates.
    
    Args:
        corners: 4x9 homogeneous coordinates of box corners
        intrinsic: 3x4 camera intrinsic matrix
        extrinsic: 4x4 lidar-to-camera transformation matrix
        
    Returns:
        projected: Nx2 array of valid projected points
        valid_mask: List of bools indicating which corners are valid (in front of camera)
    """
    # Transform to camera coordinates
    cam_coords = extrinsic @ corners
    
    # Project to image plane
    img_coords = intrinsic @ cam_coords
    
    # Check which points are in front of the camera
    valid_mask = [img_coords[2, i] > 0 for i in range(9)]
    
    # Normalize by depth
    projected = np.zeros((9, 2))
    for i in range(9):
        if valid_mask[i]:
            projected[i, 0] = img_coords[0, i] / img_coords[2, i]
            projected[i, 1] = img_coords[1, i] / img_coords[2, i]
    
    return projected, valid_mask


def calculate_iou(
    bbox_2d: Tuple[float, float, float, float],
    projected_hull: np.ndarray,
    image_shape: Tuple[int, int]
) -> float:
    """
    Calculate IoU between a 2D bounding box and a projected 3D convex hull.
    
    Args:
        bbox_2d: (x_min, y_min, x_max, y_max) of the 2D box
        projected_hull: Nx2 array of convex hull vertices
        image_shape: (height, width) of the image
        
    Returns:
        IoU value between 0 and 1
    """
    x_min, y_min, x_max, y_max = bbox_2d
    
    # Create polygons
    box_poly = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    hull_poly = Polygon(projected_hull)
    image_poly = Polygon([(0, 0), (image_shape[1], 0), 
                          (image_shape[1], image_shape[0]), (0, image_shape[0])])
    
    # Validate polygons
    if not box_poly.is_valid or not hull_poly.is_valid:
        return 0.0
    
    # Calculate intersection and union
    intersection_area = box_poly.intersection(hull_poly).area
    
    # Union considers only the part of the 3D projection within the image
    hull_in_image = hull_poly.intersection(image_poly).area
    union_area = box_poly.area + hull_in_image - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def match_2d_to_3d(
    labels_2d: List[BBox2D],
    labels_3d: List[BBox3D],
    extrinsic: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Match 2D labels to 3D labels based on object type and dimensions.
    
    Returns list of (2d_idx, 3d_idx) tuples.
    """
    matches = []
    
    for i, label_2d in enumerate(labels_2d):
        best_match = -1
        best_dist = float('inf')
        
        for j, label_3d in enumerate(labels_3d):
            # Check type match
            if label_2d.object_type != label_3d.object_type:
                continue
            
            # Check dimension match
            if (abs(label_2d.length - label_3d.length) > 0.01 or
                abs(label_2d.width - label_3d.width) > 0.01 or
                abs(label_2d.height - label_3d.height) > 0.01):
                continue
            
            # If multiple matches, use closest by position
            pos_3d = np.array([label_3d.x, label_3d.y, label_3d.z, 1.0])
            pos_cam = (extrinsic @ pos_3d)[:3]
            
            dist = np.sqrt(
                (pos_cam[0] - label_2d.loc_x)**2 +
                (pos_cam[1] - label_2d.loc_y)**2 +
                (pos_cam[2] - label_2d.loc_z)**2
            )
            
            if dist < best_dist:
                best_dist = dist
                best_match = j
        
        if best_match >= 0:
            matches.append((i, best_match))
    
    return matches


def compute_iou_for_frame(
    frame_id: int,
    camera: str,
    labels_2d: List[BBox2D],
    labels_3d: List[BBox3D],
    calib: CalibrationData,
    image_shape: Tuple[int, int],
    filter_types: Optional[List[str]] = None,
    max_occlusion: int = 40,
    max_truncation: int = 40,
) -> List[IoUResult]:
    """
    Compute IoU for all matched objects in a single frame.
    
    Args:
        frame_id: Frame number
        camera: Camera name
        labels_2d: 2D bounding box labels
        labels_3d: 3D bounding box labels for this frame
        calib: Calibration data
        image_shape: (height, width) of the image
        filter_types: Optional list of object types to include (None = all)
        max_occlusion: Max occlusion level to include
        max_truncation: Max truncation level to include
        
    Returns:
        List of IoUResult for each matched object
    """
    results = []
    
    # Match 2D to 3D labels
    matches = match_2d_to_3d(labels_2d, labels_3d, calib.extrinsic)
    
    for idx_2d, idx_3d in matches:
        label_2d = labels_2d[idx_2d]
        label_3d = labels_3d[idx_3d]
        
        # Apply filters
        if filter_types and label_3d.object_type not in filter_types:
            continue
        if label_2d.occluded > max_occlusion:
            continue
        if label_2d.truncated > max_truncation:
            continue
        
        # Skip parking lots and don't care
        if "parking" in label_3d.object_type or "dont" in label_3d.object_type:
            continue
        
        # Get 3D box corners
        corners = get_bbox_corners(label_3d)
        
        # Project to 2D
        projected, valid_mask = project_3d_to_2d(
            corners, calib.intrinsic, calib.extrinsic
        )
        
        # Check if enough corners are valid
        valid_count = sum(valid_mask)
        if valid_count < 4:
            results.append(IoUResult(
                frame_id=frame_id,
                camera=camera,
                object_type=label_3d.object_type,
                track_id=label_3d.track_id,
                iou=0.0,
                distance=np.sqrt(label_3d.x**2 + label_3d.y**2),
                bbox_2d=(label_2d.x1, label_2d.y1, label_2d.x2, label_2d.y2),
                projected_valid=False,
            ))
            continue
        
        # Get convex hull of valid projected points
        valid_points = projected[valid_mask]
        try:
            hull = ConvexHull(valid_points)
            hull_vertices = valid_points[hull.vertices]
        except Exception:
            continue
        
        # Calculate IoU
        iou = calculate_iou(
            (label_2d.x1, label_2d.y1, label_2d.x2, label_2d.y2),
            hull_vertices,
            image_shape
        )
        
        distance = np.sqrt(label_3d.x**2 + label_3d.y**2)
        
        results.append(IoUResult(
            frame_id=frame_id,
            camera=camera,
            object_type=label_3d.object_type,
            track_id=label_3d.track_id,
            iou=iou,
            distance=distance,
            bbox_2d=(label_2d.x1, label_2d.y1, label_2d.x2, label_2d.y2),
            projected_valid=True,
        ))
    
    return results


def aggregate_metrics(results: List[IoUResult]) -> IoUMetrics:
    """Compute aggregate statistics from individual IoU results."""
    if not results:
        return IoUMetrics(
            mean_iou=0.0, median_iou=0.0, std_iou=0.0,
            min_iou=0.0, max_iou=0.0, count=0, results=[]
        )
    
    ious = [r.iou for r in results if r.projected_valid]
    
    if not ious:
        return IoUMetrics(
            mean_iou=0.0, median_iou=0.0, std_iou=0.0,
            min_iou=0.0, max_iou=0.0, count=0, results=results
        )
    
    return IoUMetrics(
        mean_iou=float(np.mean(ious)),
        median_iou=float(np.median(ious)),
        std_iou=float(np.std(ious)),
        min_iou=float(np.min(ious)),
        max_iou=float(np.max(ious)),
        count=len(ious),
        results=results,
    )


def find_camera_from_folder(folder_name: str) -> Optional[str]:
    """Extract camera name from folder name like 'xIYH05-cam-02-PRIMAX-...'"""
    for cam_id, cam_name in FOLDER_TO_CAMERA.items():
        if cam_id in folder_name:
            return cam_name
    return None


def discover_sessions(data_dir: str) -> List[str]:
    """Discover all session directories in the data folder."""
    processed_dir = osp.join(data_dir, "processed", "KITTI_CAM_FRAME")
    if not osp.isdir(processed_dir):
        return []
    
    sessions = [d for d in os.listdir(processed_dir) 
                if osp.isdir(osp.join(processed_dir, d))]
    return sorted(sessions)


def discover_cameras(data_dir: str, session: str) -> List[str]:
    """Discover all camera folders for a session."""
    session_dir = osp.join(data_dir, "processed", "KITTI_CAM_FRAME", session)
    if not osp.isdir(session_dir):
        return []
    
    cameras = []
    for folder in os.listdir(session_dir):
        cam = find_camera_from_folder(folder)
        if cam:
            cameras.append(cam)
    return sorted(cameras)


def get_calib_folder_for_camera(data_dir: str, session: str, camera: str) -> Optional[str]:
    """Find the calibration folder for a specific camera."""
    calib_session_dir = osp.join(data_dir, "processed", "calib", session)
    if not osp.isdir(calib_session_dir):
        return None
    
    cam_id = CAMERA_MAP.get(camera)
    if not cam_id:
        return None
    
    for folder in os.listdir(calib_session_dir):
        if cam_id in folder and osp.isdir(osp.join(calib_session_dir, folder)):
            return osp.join(calib_session_dir, folder)
    
    return None


def get_label_folder_for_camera(data_dir: str, session: str, camera: str) -> Optional[str]:
    """Find the 2D label folder for a specific camera."""
    label_session_dir = osp.join(data_dir, "processed", "KITTI_CAM_FRAME", session)
    if not osp.isdir(label_session_dir):
        return None
    
    cam_id = CAMERA_MAP.get(camera)
    if not cam_id:
        return None
    
    for folder in os.listdir(label_session_dir):
        if cam_id in folder and osp.isdir(osp.join(label_session_dir, folder)):
            return osp.join(label_session_dir, folder)
    
    return None


def compute_session_iou(
    data_dir: str,
    session: str,
    calib_dir: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    filter_types: Optional[List[str]] = None,
    frames: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, IoUMetrics]:
    """
    Compute IoU metrics for an entire session.
    
    Args:
        data_dir: Root data directory
        session: Session name
        calib_dir: Optional override directory for calibration files
        cameras: Optional list of cameras to process (None = all)
        filter_types: Optional list of object types to include
        frames: Optional list of frame IDs to process (None = all)
        verbose: Print progress
        
    Returns:
        Dict mapping camera name to IoUMetrics
    """
    # Load 3D labels
    label_3d_file = osp.join(data_dir, "processed", "KITTI_SENSORFUSION", f"{session}.txt")
    if not osp.exists(label_3d_file):
        print(f"Warning: 3D label file not found: {label_3d_file}")
        return {}
    
    labels_3d_by_frame = load_3d_labels(label_3d_file)
    
    # Discover cameras if not specified
    if cameras is None:
        cameras = discover_cameras(data_dir, session)
    
    results_by_camera: Dict[str, IoUMetrics] = {}
    
    for camera in cameras:
        if verbose:
            print(f"  Processing camera: {camera}")
        
        # Get folder paths
        if calib_dir:
            calib_folder = get_calib_folder_for_camera(calib_dir, session, camera)
        else:
            calib_folder = get_calib_folder_for_camera(data_dir, session, camera)
        
        label_folder = get_label_folder_for_camera(data_dir, session, camera)
        
        if not calib_folder or not label_folder:
            print(f"    Warning: Could not find folders for {camera}")
            continue
        
        # Get image shape for this camera
        image_shape = IMAGE_SHAPES.get(camera, (2160, 3840))
        
        # Process each frame
        all_results: List[IoUResult] = []
        
        # List calibration files to determine frames
        calib_files = sorted([f for f in os.listdir(calib_folder) if f.endswith('.txt')])
        
        for calib_file in calib_files:
            # Extract frame number from filename
            # Format: xIYH05-cam-02-PRIMAX-IMX728-20250828_221259-000100.txt
            try:
                frame_str = calib_file.split('-')[-1].replace('.txt', '')
                frame_id = int(frame_str)
            except (ValueError, IndexError):
                continue
            
            # Filter by frame if specified
            if frames is not None and frame_id not in frames:
                continue
            
            # Load calibration
            calib_path = osp.join(calib_folder, calib_file)
            try:
                calib = load_calibration(calib_path)
            except Exception as e:
                print(f"    Warning: Failed to load calibration {calib_file}: {e}")
                continue
            
            # Load 2D labels (same filename pattern in label folder)
            label_file = calib_file  # Same naming convention
            label_path = osp.join(label_folder, label_file)
            labels_2d = load_2d_labels(label_path)
            
            # Get 3D labels for this frame
            labels_3d = labels_3d_by_frame.get(frame_id, [])
            
            if not labels_2d or not labels_3d:
                continue
            
            # Compute IoU for this frame
            frame_results = compute_iou_for_frame(
                frame_id=frame_id,
                camera=camera,
                labels_2d=labels_2d,
                labels_3d=labels_3d,
                calib=calib,
                image_shape=image_shape,
                filter_types=filter_types,
            )
            
            all_results.extend(frame_results)
        
        # Aggregate metrics
        metrics = aggregate_metrics(all_results)
        results_by_camera[camera] = metrics
        
        if verbose:
            print(f"    Mean IoU: {metrics.mean_iou:.4f} ({metrics.count} objects)")
    
    return results_by_camera


def save_results_csv(
    results: Dict[str, Dict[str, IoUMetrics]],
    output_path: str,
    include_details: bool = False,
):
    """
    Save IoU results to CSV file.
    
    Args:
        results: Dict mapping session -> camera -> IoUMetrics
        output_path: Output CSV path
        include_details: If True, include per-object results
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write summary
        writer.writerow(['session', 'camera', 'mean_iou', 'median_iou', 'std_iou', 
                        'min_iou', 'max_iou', 'count'])
        
        for session, cameras in results.items():
            for camera, metrics in cameras.items():
                writer.writerow([
                    session, camera, 
                    f"{metrics.mean_iou:.4f}",
                    f"{metrics.median_iou:.4f}",
                    f"{metrics.std_iou:.4f}",
                    f"{metrics.min_iou:.4f}",
                    f"{metrics.max_iou:.4f}",
                    metrics.count,
                ])
        
        if include_details:
            writer.writerow([])
            writer.writerow(['--- DETAILED RESULTS ---'])
            writer.writerow(['session', 'camera', 'frame_id', 'track_id', 
                           'object_type', 'iou', 'distance', 'valid'])
            
            for session, cameras in results.items():
                for camera, metrics in cameras.items():
                    for r in metrics.results:
                        writer.writerow([
                            session, camera, r.frame_id, r.track_id,
                            r.object_type, f"{r.iou:.4f}", 
                            f"{r.distance:.2f}", r.projected_valid,
                        ])
    
    print(f"Results saved to: {output_path}")


def save_results_json(
    results: Dict[str, Dict[str, IoUMetrics]],
    output_path: str,
):
    """Save IoU results to JSON file."""
    output = {}
    
    for session, cameras in results.items():
        output[session] = {}
        for camera, metrics in cameras.items():
            output[session][camera] = {
                'mean_iou': metrics.mean_iou,
                'median_iou': metrics.median_iou,
                'std_iou': metrics.std_iou,
                'min_iou': metrics.min_iou,
                'max_iou': metrics.max_iou,
                'count': metrics.count,
            }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU metrics for calibration quality evaluation"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Root data directory containing processed/ folder"
    )
    parser.add_argument(
        "--calib_dir", type=str, default=None,
        help="Optional override directory for calibration files (for comparing different calibrations)"
    )
    parser.add_argument(
        "--output", type=str, default="iou_results.csv",
        help="Output file path (.csv or .json)"
    )
    parser.add_argument(
        "--sessions", type=str, nargs="*", default=None,
        help="Specific sessions to process (default: all)"
    )
    parser.add_argument(
        "--cameras", type=str, nargs="*", default=None,
        choices=list(CAMERA_MAP.keys()),
        help="Specific cameras to process (default: all)"
    )
    parser.add_argument(
        "--filter_types", type=str, nargs="*", default=None,
        help="Object types to include (e.g., 'vehicle.small_vehicle.sedan')"
    )
    parser.add_argument(
        "--frames", type=int, nargs="*", default=None,
        help="Specific frame IDs to process (e.g., --frames 1 10 50)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Include per-object detailed results in output"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Discover sessions
    if args.sessions:
        sessions = args.sessions
    else:
        sessions = discover_sessions(args.data_dir)
    
    if not sessions:
        print(f"No sessions found in {args.data_dir}")
        return
    
    print(f"Found {len(sessions)} session(s)")
    
    # Process each session
    all_results: Dict[str, Dict[str, IoUMetrics]] = {}
    
    for session in sessions:
        if not args.quiet:
            print(f"\nProcessing session: {session}")
        
        session_results = compute_session_iou(
            data_dir=args.data_dir,
            session=session,
            calib_dir=args.calib_dir,
            cameras=args.cameras,
            filter_types=args.filter_types,
            frames=args.frames,
            verbose=not args.quiet,
        )
        
        if session_results:
            all_results[session] = session_results
    
    # Save results
    if args.output.endswith('.json'):
        save_results_json(all_results, args.output)
    else:
        save_results_csv(all_results, args.output, include_details=args.detailed)
    
    # Print summary
    print("\n=== SUMMARY ===")
    total_count = 0
    total_iou_sum = 0.0
    
    for session, cameras in all_results.items():
        print(f"\n{session}:")
        for camera, metrics in cameras.items():
            print(f"  {camera}: mean={metrics.mean_iou:.4f}, "
                  f"median={metrics.median_iou:.4f}, count={metrics.count}")
            total_count += metrics.count
            total_iou_sum += metrics.mean_iou * metrics.count
    
    if total_count > 0:
        overall_mean = total_iou_sum / total_count
        print(f"\nOverall mean IoU: {overall_mean:.4f} ({total_count} total objects)")


if __name__ == "__main__":
    main()
