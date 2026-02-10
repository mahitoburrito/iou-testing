#!/usr/bin/env python3
"""
Visualize IoU comparison between two calibrations on a single frame.

Draws 2D annotation boxes (green) and projected 3D convex hulls for
the original calibration (blue) and an alternative calibration (red)
on the camera image, with per-object IoU labels.
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# Re-use core logic from compute_iou
from compute_iou import (
    CAMERA_MAP,
    IMAGE_SHAPES,
    load_calibration,
    load_3d_labels,
    load_2d_labels,
    get_bbox_corners,
    project_3d_to_2d,
    calculate_iou,
    match_2d_to_3d,
)


def draw_hull(ax, vertices, color, label=None, linewidth=2):
    """Draw a convex hull polygon outline on an axes."""
    poly = plt.Polygon(vertices, fill=False, edgecolor=color,
                       linewidth=linewidth, linestyle="-", label=label)
    ax.add_patch(poly)


def visualize_frame(
    image_path,
    labels_2d,
    labels_3d,
    calib_orig,
    calib_new,
    camera,
    frame_id,
    output_path,
):
    img = Image.open(image_path)
    image_shape = IMAGE_SHAPES.get(camera, (img.height, img.width))

    # Match 2D to 3D using original calib (same matching for both)
    matches = match_2d_to_3d(labels_2d, labels_3d, calib_orig.extrinsic)

    fig, ax = plt.subplots(1, 1, figsize=(24, 14))
    ax.imshow(img)
    ax.set_title(f"{camera} — Frame {frame_id}  |  Original (blue) vs calibAnything (red) vs 2D GT (green)",
                 fontsize=14, fontweight="bold")

    legend_entries = []

    for idx_2d, idx_3d in matches:
        l2 = labels_2d[idx_2d]
        l3 = labels_3d[idx_3d]

        if "parking" in l3.object_type or "dont" in l3.object_type:
            continue

        bbox_2d = (l2.x1, l2.y1, l2.x2, l2.y2)
        distance = np.sqrt(l3.x ** 2 + l3.y ** 2)

        # ---- 2D GT box (green) ----
        rect = plt.Rectangle((l2.x1, l2.y1), l2.x2 - l2.x1, l2.y2 - l2.y1,
                              fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)

        corners = get_bbox_corners(l3)

        # ---- Original calibration projection (blue) ----
        proj_orig, mask_orig = project_3d_to_2d(corners, calib_orig.intrinsic, calib_orig.extrinsic)
        iou_orig = 0.0
        if sum(mask_orig) >= 4:
            valid_pts = proj_orig[mask_orig]
            try:
                hull = ConvexHull(valid_pts)
                hull_verts = valid_pts[hull.vertices]
                draw_hull(ax, hull_verts, color="deepskyblue", linewidth=2)
                iou_orig = calculate_iou(bbox_2d, hull_verts, image_shape)
            except Exception:
                pass

        # ---- New calibration projection (red) ----
        proj_new, mask_new = project_3d_to_2d(corners, calib_new.intrinsic, calib_new.extrinsic)
        iou_new = 0.0
        if sum(mask_new) >= 4:
            valid_pts = proj_new[mask_new]
            try:
                hull = ConvexHull(valid_pts)
                hull_verts = valid_pts[hull.vertices]
                draw_hull(ax, hull_verts, color="red", linewidth=2)
                iou_new = calculate_iou(bbox_2d, hull_verts, image_shape)
            except Exception:
                pass

        # ---- Label ----
        short_type = l3.object_type.split(".")[-1]
        label_text = (f"T{l3.track_id} {short_type} ({distance:.0f}m)\n"
                      f"orig={iou_orig:.3f}  new={iou_new:.3f}")

        # Place label above the 2D box
        ax.text(l2.x1, l2.y1 - 10, label_text,
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                verticalalignment="bottom")

    # Legend
    legend_handles = [
        mpatches.Patch(edgecolor="lime", facecolor="none", linewidth=2, label="2D Ground Truth"),
        mpatches.Patch(edgecolor="deepskyblue", facecolor="none", linewidth=2, label="Original Calib (projected 3D)"),
        mpatches.Patch(edgecolor="red", facecolor="none", linewidth=2, label="calibAnything (projected 3D)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              facecolor="black", edgecolor="white", labelcolor="white")

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    data_dir = "./data"
    session = "xIYH05-gravity-USA-vin116-20250828_221259"
    camera = "FNC"
    cam_folder = "cam-03"
    frame_id = 1

    # Paths
    cam_dir_name = f"xIYH05-{cam_folder}-PRIMAX-IMX728-20250828_221259"
    image_path = os.path.join(data_dir, cam_dir_name, "png_files",
                              f"{cam_dir_name}-000001.png")

    calib_orig_path = os.path.join(data_dir, "processed", "calib", session,
                                   cam_dir_name, f"{cam_dir_name}-000001.txt")
    calib_new_path = os.path.join("lucid_calib_test", "processed", "calib", session,
                                  cam_dir_name, f"{cam_dir_name}-000001.txt")

    label_3d_path = os.path.join(data_dir, "processed", "KITTI_SENSORFUSION",
                                 f"{session}.txt")
    label_2d_path = os.path.join(data_dir, "processed", "KITTI_CAM_FRAME", session,
                                 cam_dir_name, f"{cam_dir_name}-000001.txt")

    # Load data
    calib_orig = load_calibration(calib_orig_path)
    calib_new = load_calibration(calib_new_path)
    labels_3d_all = load_3d_labels(label_3d_path)
    labels_3d = labels_3d_all.get(frame_id, [])
    labels_2d = load_2d_labels(label_2d_path)

    print(f"Image: {image_path}")
    print(f"2D labels: {len(labels_2d)}, 3D labels for frame {frame_id}: {len(labels_3d)}")

    visualize_frame(
        image_path=image_path,
        labels_2d=labels_2d,
        labels_3d=labels_3d,
        calib_orig=calib_orig,
        calib_new=calib_new,
        camera=camera,
        frame_id=frame_id,
        output_path="cam03_frame1_iou_comparison.png",
    )


if __name__ == "__main__":
    main()
