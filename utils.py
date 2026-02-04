from os import path as osp
import os
from addict import Dict as Addict
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import yaml
import multiprocessing
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    Args:
        qw (float): Real part of the quaternion.
        qx (float): i-component of the quaternion.
        qy (float): j-component of the quaternion.
        qz (float): k-component of the quaternion.
    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    # Normalize the quaternion to ensure it's a unit quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    # Calculate the components of the rotation matrix
    R11 = 1 - 2*(qy**2 + qz**2)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)

    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx**2 + qz**2)
    R23 = 2*(qy*qz - qx*qw)

    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx**2 + qy**2)

    return np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])

def apply_calibration_adjustment(calib_matrix, dx, dy, dz, qw, qx, qy, qz):
    """
    Applies a translation (dx, dy, dz) and rotation (qw, qx, qy, qz) 
    adjustment to a 4x4 calibration matrix.

    Args:
        calib_matrix (np.ndarray): The 4x4 homogeneous transformation matrix
                                   representing the current calibration.
        dx (float): Translation along the x-axis.
        dy (float): Translation along the y-axis.
        dz (float): Translation along the z-axis.
        qw (float): Real part of the quaternion.
        qx (float): i-component of the quaternion.
        qy (float): j-component of the quaternion.
        qz (float): k-component of the quaternion.

    Returns:
        np.ndarray: The updated 4x4 calibration matrix.
    """

    # 1. Convert the adjustment quaternion to a rotation matrix using the custom function
    rotation_adjustment_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    # 2. Create the adjustment homogeneous transformation matrix
    translation_adjustment_vector = np.array([dx, dy, dz])

    adjustment_matrix = np.eye(4)
    adjustment_matrix[:3, :3] = rotation_adjustment_matrix
    adjustment_matrix[:3, 3] = translation_adjustment_vector

    # 3. Apply the adjustment (assuming post-multiplication as discussed before)
    # new_calib_matrix = calib_matrix @ adjustment_matrix
    new_calib_matrix = np.linalg.inv(adjustment_matrix) @ calib_matrix

    return new_calib_matrix

def get_gt_3d(gt_path):

    gt = dict()

    with open(gt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(" ")
            # 0 timestamp
            # 1 frame_id
            # 2 track_id
            # 3 object_type
            # 4 truncated
            # 5 occluded
            # 6 heading
            # 7-9 length width height
            # 10-12 location x y z
            # 13 rotation_z
            # 14 tja
            # 15 lane_id
            # 16 brake_light
            # 17 left_light
            # 18 right_light
            if "\n" in data[-1]:
                data[-1] = data[-1].replace("\n", "")
            if data[1] not in gt:
                gt[data[1]] = [data]
            else:
                gt[data[1]].append(data)
    
    return gt

def get_gt_2d(label_image_folder, label_files):
    labels = []
    for label_file in label_files:
        label = []
        cur_file = osp.join(label_image_folder, label_file)
        with open(cur_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(" ")
                if len(data) < 8:
                    continue
                if "\n" in data[-1]:
                    data[-1] = data[-1].replace("\n", "")
                label.append(data)
        labels.append(label)
    return labels

def get_calib(calib_file_path):
    with open(calib_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(" ")
            if "P0:" in line:
                intrinsic = list(map(float, data[1:]))
                intrinsic = np.array(intrinsic).reshape(3, 4)
            if "Tr_velo_to_cam:" in line:
                extrinsic = list(map(float, data[1:]))
                extrinsic = np.array(extrinsic + [0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
    return intrinsic, extrinsic

def get_bbox_corners(bbox):
    # bbox: [length, width, height, x, y, z, heading]
    center = np.array(bbox[3:6] + [1.0])
    size = np.array(bbox[:3])
    heading = bbox[6]
    # heading = bbox[0]
    corners = np.ones(shape=(4,9))
    corners[0, :] = center[0]
    corners[1, :] = center[1]
    corners[2, :] = center[2]

    corners[0, 0:4] += size[0]/2 * np.cos(heading)
    corners[0, 4:8] -= size[0]/2 * np.cos(heading)
    corners[1, 0:4] += size[0]/2 * np.sin(heading)
    corners[1, 4:8] -= size[0]/2 * np.sin(heading)

    corners[1, 0:2] += size[1]/2 * np.cos(heading)
    corners[1, 2:4] -= size[1]/2 * np.cos(heading)
    corners[1, 4:6] += size[1]/2 * np.cos(heading)
    corners[1, 6:8] -= size[1]/2 * np.cos(heading)

    corners[0, 0:2] -= size[1]/2 * np.sin(heading)
    corners[0, 2:4] += size[1]/2 * np.sin(heading)
    corners[0, 4:6] -= size[1]/2 * np.sin(heading)
    corners[0, 6:8] += size[1]/2 * np.sin(heading)

    corners[2, 0:8:2] += size[2]/2
    corners[2, 1:8:2] -= size[2]/2

    return corners

def draw_bbox(image, corners, corners_valid, color=(0, 255, 0)):
    lines = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    for line in lines:
        if corners_valid[line[0]] and corners_valid[line[1]]:
            cv2.line(image, tuple(corners[line[0], :2]), tuple(corners[line[1], :2]), color, 2)
    if corners_valid[8]:
        cv2.circle(image, corners[8, :2], 5, color, -1)

camera_map = Addict()
camera_map.FNC = "cam-03"
camera_map.FWC_C = "cam-02"
camera_map.FWC_L = "cam-07"
camera_map.FWC_R = "cam-05"
camera_map.RNC_C = "cam-06"
camera_map.RNC_L = "cam-08"
camera_map.RNC_R = "cam-04"

def show_image(cam, image_path, intrinsic, extrinsic, labels, label_2d, show_2d=True, show_3d=False, reverse=False, resize=False):
    # print("intrinsic:", intrinsic)
    # print("extrinsic:", extrinsic)
    # print (cam, resize)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2d labels
    # 0 object_type
    # 1 truncated
    # 2 occluded
    # 3 heading
    # 4 x1
    # 5 y1
    # 6 x2
    # 7 y2
    # 8 height
    # 9 width
    # 10 length
    # 11 location_x
    # 12 location_y
    # 13 location_z
    # 14 rotation_z
    # 15 tja
    # 16 lane_id
    # 17 brake_light
    # 18 left_light
    # 19 right_light

    # draw 2d labels
    if show_2d:
        with open(label_2d, "r") as f:
            lines = f.readlines()
            i = 1
            for line in lines:
                # if "RNC_C" in cam:
                #     print(line)
                data = line.split(" ")
                if len(data) < 8:
                    continue
                x1, y1, x2, y2 = map(float, data[4:8])
                # if i != 3:
                #     i += 1
                #     continue
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(image, str(i), (int(x1)+10, int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                # # cv2.putText(image, f"{i} {data[0]}", (int(x1)+10, int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(image, f"{i} Trunc {data[1]}, Occ {data[2]}", (30, i*40+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                # cv2.putText(image, f"{i} Trunc {data[1]}, Occ {data[2]}, x1 {x1}, y1 {y1}", (30, i*40+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                i+=1

    # draw 3d labels
    if show_3d:
        # with open(label_2d, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         data = line.split(" ")
        #         bbox_3d = [float(data[3]), float(data[10]), float(data[9]), float(data[8]), float(data[13]), -float(data[11]), -float(data[12])+float(data[8])/2, float(data[14])-np.pi/2]
        #         corners = get_bbox_corners(bbox_3d, reverse)
        #         transform = np.array([[0, -1, 0, 0],
        #                               [0, 0, -1, 0],
        #                               [1, 0, 0, 0],
        #                               [0, 0, 0, 1]])
        #         corners = np.matmul(transform, corners)
        #         corners = np.matmul(intrinsic, corners)
        #         corners_valid = [False]*9
        #         for i in range(9):
        #             if corners[2, i] > 0:
        #                 corners_valid[i] = True
        #                 corners[:2, i] /= corners[2, i]
        #         corners = corners.T.astype(int)

        #         # draw lines between corners
        #         draw_bbox(image, corners, corners_valid,color=(0, 0, 255))

        ii = 1
        for label in labels:
            # if label[-1] != 15.0 and label[-1] != 72.0:
            #     continue
            corners = get_bbox_corners(label, reverse)
            corners = np.matmul(extrinsic, corners)
            corners = np.matmul(intrinsic, corners)
            corners_valid = [False]*9
            for i in range(9):
                if corners[2, i] > 0:
                    corners_valid[i] = True
                    corners[:2, i] /= corners[2, i]
            if resize:
                corners[:2, :] /= 1.2
            corners = corners.T.astype(int)

            # draw lines between corners
            draw_bbox(image, corners, corners_valid)

            # min_x = min(corners[:, 0])
            # min_y = min(corners[:, 1])
            if corners_valid[8]:
                cv2.putText(image, str(int(label[-2])), (int(corners[-1,0])+10, int(corners[-1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(image, f"track id: {int(label[-2])} Occ {label[-1]}", (1030, ii*40+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                ii += 1
        # for label in labels_comp:
        #     corners = get_bbox_corners(label, reverse)
        #     corners = np.matmul(extrinsic, corners)
        #     corners = np.matmul(intrinsic, corners)
        #     corners_valid = [False]*9
        #     for i in range(9):
        #         if corners[2, i] > 0:
        #             corners_valid[i] = True
        #             corners[:2, i] /= corners[2, i]
        #     corners = corners.T.astype(int)

        #     # draw lines between corners
        #     draw_bbox(image, corners, corners_valid, color=(0, 0, 255))


    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # plt.title(cam)
    # plt.show()
    return image, cam

def get_cam_images(data_path, cam):
    cam_images = []
    cam_timestamps = []
    cam_dirs = os.listdir(data_path)

    cam_dir = [cam_folder for cam_folder in cam_dirs if camera_map[cam] in cam_folder][0]
    image_folder = osp.join(data_path, cam_dir, "png_files")
    timestamp_file = osp.join(data_path, cam_dir, "timestamps.txt")
    with open(timestamp_file, "r") as f:
        timestamps = f.readlines()
        for timestamp in timestamps:
            cam_tiemstamp = int(float(timestamp))%1000000 / 1000
            cam_timestamps.append(cam_tiemstamp)
    image_files = os.listdir(image_folder)
    image_files = [i for i in image_files if ".png" in i]
    image_files.sort()
    for image_file in image_files:
        image_path = osp.join(image_folder, image_file)
        cam_images.append(image_path)
    if len(cam_images) == 0:
        print(f"No images found for {cam} in {data_path}")
        return None
    if len(cam_images) != len(cam_timestamps):
        print(f"Different number of images and timestamps for {cam} in {data_path}: {len(cam_images)} vs {len(cam_timestamps)}")
        return None
    return {"images": cam_images, "timestamps": cam_timestamps}

def get_calib_data(batch_path, session, cams):
    calib_path = osp.join(batch_path, "calibs")
    if not osp.isdir(calib_path):
        calib_path = osp.join(batch_path, "calib")
    if not osp.isdir(calib_path):
        label_path = osp.join(batch_path, "labels")
        label_folders = os.listdir(label_path)

        try:
            label_folders = [i for i in label_folders if "sf" in i or "sensorfusion" in i][0]
            calib_path = osp.join(label_path, label_folders, "calib")
        except IndexError:
            calib_path = osp.join(label_path, "calib")
            label_folders = "./"

    session_calib_path = osp.join(calib_path, session)
    calib_cam_dirs = os.listdir(session_calib_path)
    calibs = {}
    
    offset_path = osp.join(session_calib_path, "offsets")
    for cam in cams:
        try:
            calib_cam_dir = [cam_dir for cam_dir in calib_cam_dirs if camera_map[cam] in cam_dir][0]
            calib_file_path = osp.join(session_calib_path, calib_cam_dir)
            calib_files = os.listdir(calib_file_path)
            calib_file = [i for i in calib_files if ".txt" in i][0]
            intrinsic, extrinsic = get_calib(osp.join(calib_file_path, calib_file))
            calibs[cam] = dict()
            calibs[cam]["intrinsic"] = intrinsic
            calibs[cam]["extrinsic"] = extrinsic

        except Exception:
            print(f"Error getting calibration data for {cam} in {session}")
            return None
    
        if osp.isdir(offset_path):
            offset_files = os.listdir(offset_path)
            for offset_file in offset_files:
                if camera_map[cam] in offset_file:
                    offset_file_path = osp.join(offset_path, offset_file)
                    # print('offset file path:', offset_file_path)
                    with open(offset_file_path, "r") as f:
                        offset_data = yaml.safe_load(f)
                        # print(calibs[cam]["extrinsic"])
                        calibs[cam]["extrinsic"] = apply_calibration_adjustment(
                            calibs[cam]["extrinsic"],
                            offset_data[cam]["dx"],
                            offset_data[cam]["dy"],
                            offset_data[cam]["dz"],
                            offset_data[cam]["qw"],
                            offset_data[cam]["qx"],
                            offset_data[cam]["qy"],
                            offset_data[cam]["qz"]
                        )
    return calibs

def get_session_data(batch_path, session, cams, load_images=True):
    data = {}
    data_path = osp.join(batch_path, "data")
    label_path = osp.join(batch_path, "labels")

    session_path = osp.join(data_path, session)

    images = {}
    labels = {}

    if load_images:
        # print("getting image data")
        # get images data
        with multiprocessing.Pool(processes=len(cams)) as pool:
            results = pool.starmap(get_cam_images, [(session_path, cam) for cam in cams])
        for cam, result in zip(cams, results):
            images[cam] = result

        for cam in cams:
            if images[cam] is None:
                print(f"No images found for {cam} in {session}")
                return None
            
        max_frames = len(images[cams[0]])
        for cam in cams:
            if len(images[cam]) != max_frames:
                print(f"Different number of frames for {cam} in {session}: {len(images[cam])} vs {max_frames}")
            if len(images[cam]) < max_frames:
                max_frames = len(images[cam])
        if max_frames == 0:
            print(f"No images found in {session}")
            return None
        
        data["images"] = images

    # print("getting calibration data")
    # get calibration data
    calibs = get_calib_data(batch_path, session, cams)
    if calibs is None:
        print(f"No calibration data found for {session}")
        return None
    
    data["calibs"] = calibs
    data["image_path"] = session_path

    # print("getting labels data")
    # get labels data
    # get 3d labels
    label_folders = os.listdir(label_path)

    try:
        label_folders = [i for i in label_folders if "sf" in i or "sensorfusion" in i][0]
        label_folder = osp.join(label_path, label_folders, "KITTI_SENSORFUSION")
    except IndexError:
        label_folder = osp.join(label_path, "KITTI_SENSORFUSION")
        label_folders = "./"
    label_sessions = os.listdir(label_folder)
    selected_label_session = session + '.txt'
    label_session_path = osp.join(label_folder, selected_label_session)
    label_3d = get_gt_3d(label_session_path)
    if label_3d is None:
        print(f"No 3d labels found for {session}")
        return None
    labels["3d"] = label_3d

    # get 2d labels
    label_2d = {}
    label_folder_2d = osp.join(label_path, label_folders, "KITTI_CAM_FRAME")
    label_session_2d_path = osp.join(label_folder_2d, session)
    label_cam_dirs = os.listdir(label_session_2d_path)
    for cam in cams:
        label_cam_dir = [cam_folder for cam_folder in label_cam_dirs if camera_map[cam] in cam_folder or cam in cam_folder][0]
        label_image_folder = osp.join(label_session_2d_path, label_cam_dir)
        label_files = os.listdir(label_image_folder)
        label_files = [i for i in label_files if ".txt" in i]
        label_files.sort()
        label_2d[cam] = get_gt_2d(label_image_folder, label_files)
    labels["2d"] = label_2d
    data["labels"] = labels

    return data


def get_data(batch_path, cams, load_images=True):
    data_path = osp.join(batch_path, "data")
    sessions = os.listdir(data_path)
    if len(sessions) == 0:
        return None
    sessions = [i for i in sessions if osp.isdir(osp.join(data_path, i))]
    data = {}
    for session in sessions:
        session_name = session[:6]

        session_data = get_session_data(batch_path, session, cams, load_images)
        if session_data is not None:
            data[session_name] = session_data
    
    if len(data.keys()) == 0:
        print(f"No data found in {batch_path}")
        return None
    return data

def get_batch_folders(configs):

    root_path = configs["root_path"]

    batch_candidate = os.listdir(root_path)
    batch = configs['batch_folder']
    batch_confirmed = []
    
    if len(batch) == 0:
        batch_confirmed = [batch_id for batch_id in batch if osp.isdir(osp.join(root_path, batch_id))]
    
    else:
        for keyward in batch:
            for cur_batch in batch_candidate:
                if keyward in cur_batch:
                    if osp.isdir(osp.join(root_path, cur_batch)):
                        batch_confirmed.append(cur_batch)
    
    bathc_return = []

    for batch_id in batch_confirmed:
        batch_path = osp.join(root_path, batch_id)
        label_path = osp.join(batch_path, "labels")
        if osp.isdir(label_path):
            folders = os.listdir(label_path)
            contains_sf = False
            for folder in folders:
                if "sf_bev" in folder:
                    contains_sf = True
                    break
            if contains_sf:
                bathc_return.append(batch_id)
    
    return bathc_return
    



def load_data(configs, load_images=True):
    root_path = configs["root_path"]

    batch_candidate = os.listdir(root_path)
    batch = configs['batch_folder']
    batch_confirmed = []
    
    batch_confirmed = get_batch_folders(configs)

    data = {}
    for batch_id in batch_confirmed:
        batch_path = osp.join(root_path, batch_id)
        
        batch_data = get_data(batch_path, configs['cams'], load_images)
        if batch_data is not None:
            data[batch_id] = batch_data
    
    if len(data.keys()) == 0:
        print(f"No data found in {root_path}")
        return None
    
    return data


def calculate_iou(bbox_2d, convex_hull_3d, image_shape):
    """
    Calculates the IoU between a 2D bounding box and a 2D polygon.

    Args:
        bbox_2d (list or np.array): [x_min, y_min, x_max, y_max] of the 2D box.
        convex_hull_3d (np.array): Nx2 numpy array of the convex hull vertices.

    Returns:
        float: The IoU value.
    """
    # Create Shapely Polygon objects
    x_min, y_min, x_max, y_max = bbox_2d
    box_2d_poly = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    hull_3d_poly = Polygon(convex_hull_3d)
    image_box_poly = Polygon([(0, 0), (image_shape[1], 0), (image_shape[1], image_shape[0]), (0, image_shape[0])])

    # Check for empty or invalid polygons
    if not box_2d_poly.is_valid or not hull_3d_poly.is_valid:
        return 0.0

    # Calculate intersection area
    intersection_area = box_2d_poly.intersection(hull_3d_poly).area

    # Calculate union area
    union_area = box_2d_poly.area + hull_3d_poly.intersection(image_box_poly).area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

object_type = set([
    "vehicle.large_vehicle.bus",
    "vehicle.large_vehicle.vehicle_transporter",
    "vehicle.large_vehicle.truck",
    "vehicle.large_vehicle.van",
    "vehicle.large_vehicle",
    "vehicle.large_vehicle.emergency.ambulance",
    "vehicle.large_vehicle.emergency.fire_truck",
    "vehicle.large_vehicle.emergency.school_bus",
    "vehicle.large_vehicle.construction",
    "vehicle.medium_vehicle.van",
    "vehicle.medium_vehicle.emergency.ambulance",
    "vehicle.medium_vehicle.SUV",
    "vehicle.medium_vehicle.emergency.fire_truck",
    "vehicle.medium_vehicle.pick_up",
    "vehicle.medium_vehicle.emergency.police",
    "vehicle.medium_vehicle",
    "vehicle.medium_vehicle.lucid",
    "vehicle.small_vehicle.sedan",
    "vehicle.small_vehicle.emergency.police",
    "vehicle.small_vehicle.mini_car",
    "vehicle.small_vehicle.construction",
    "vehicle.small_vehicle.hatchback",
    "vehicle.small_vehicle.lucid",
    "vehicle.small_vehicle",
    "vehicle.vehicle_with_tow",
    "vehicle.vehicle_with_tow.towed_object.trailer.boat",
    "vehicle.vehicle_with_tow.towed_object",
    "vehicle.vehicle_with_tow.towed_object.trailer.semi_trailer",
    "vehicle.vehicle_with_tow.towed_object.trailer.vehicle_bed",
    "vehicle.vehicle_with_tow.towed_object.traffic_sign",
    "vehicle.motorcycle",
    "vehicle.train.train",
    "vehicle.bicycle",
    "vehicle.train.light_rail",
    "human.pedestrian",
    "human.attached_objects.stroller",
    "human.pedestrian.with_objects",
    "human.attached_objects.shopping_cart",
    "human.pedestrian.rider",
    "human.attached_objects.hand_truck",
    "human.attached_objects",
    "movable_object.traffic_cone",
    "movable_object.trash_can",
    "movable_object.barricade",
    "movable_object.dumpster",
    "animal",
    "animal.small_animal",
    "animal.horse",
    "animal.large_animal",
    "parking_lot",
    "dont_care"

])