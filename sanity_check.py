import yaml
import numpy as np
from scipy.spatial import ConvexHull
import cv2
from shapely.geometry import Polygon

import ds.utils as utils
import os
import os.path as osp
import logging
import argparse
import datetime
from pytz import timezone

from s3_to_local_download import s3_to_local_download, upload_to_s3, delete_local_dir, get_batch_passing_sessions, write_xcom_file, \
    get_filtered_s3_to_hpc_json, add_sanity_check_logs_to_s3_to_hpc_json
### 3d labels ###
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

### 2d labels ###
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
# 14 rotation_y
# 15 tja
# 16 lane_id
# 17 brake_light
# 18 left_light
# 19 right_light

image_shape_1 = {
    "FNC" : (1800, 3200),
    "FWC_C" : (2382, 6400),
    "FWC_R" : (2472, 3840),
    "RNC_R" : (1800, 3200),
    "RNC_C" : (1800, 3200),
    "RNC_L" : (1800, 3200),
    "FWC_L" : (2472, 3840),
}

image_shape_2 = {
    "FNC" : (2160, 3840),
    "FWC_C" : (2856, 7680),
    "FWC_R" : (2472, 3840),
    "RNC_R" : (1800, 3200),
    "RNC_C" : (1800, 3200),
    "RNC_L" : (1800, 3200),
    "FWC_L" : (2472, 3840),
}

image_shape = [image_shape_1, image_shape_2]

def main():
    #parse args to save log and flag of save image
    parser = argparse.ArgumentParser(description="Sanity check for 2D and 3D labels")
    parser.add_argument("--log", type=str, default="", help="log file to save the results")
    parser.add_argument("--log_session", action="store_true", help="store log by session")
    parser.add_argument("--save_image", action="store_true", help="flag to save the images with 2D and 3D boxes")
    parser.add_argument("--result_path", type=str, default="", help="folder to save the results")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Config file to use")
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)

    assert len(configs["distance"]) == len(configs["iou_threshold"]) + 1
    
    if args.log_session:
        configs["log_session"] = True
    
    if args.save_image:
        configs["save_image"] = True

    if args.log == "":
        results_file = "results.log"
    else:
        results_file = args.log + '.log'
    
    if args.result_path != "":
        configs["result_path"] = args.result_path

    current_time = datetime.datetime.now()
    current_time = current_time.astimezone(timezone('US/Pacific')).strftime("%Y-%m-%d %H:%M:%S")

    if configs["result_path"] != "":
        result_folder = osp.join("./", "results", configs["result_path"])
    else:
        # use date and month as default folder name
        result_folder = osp.join("./", "results", datetime.datetime.now().strftime("%Y_%m_%d"))
    os.makedirs(result_folder, exist_ok=True)
    results_file = osp.join(result_folder, results_file)


    with open(results_file, "a") as f:
        f.write(f"\n\nSanity check log - {current_time}\n")
    
    logger = logging.getLogger('sanity_check')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')

    if not configs["log_session"]:
        log_handler = logging.FileHandler(configs["log_file"])
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
        # log current date/time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.astimezone(timezone('US/Pacific')).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Sanity check log - {current_time}")
        logger.info(f"Distance intervals for iou checking: {configs['distance']}")
        logger.info(f"IoU threshold for RNC_C: {[float(f'{iou_t - 0.05:.2f}') for iou_t in configs['iou_threshold']]}, IoU threshold for other cams:{configs['iou_threshold']}")
        logger.info(f"Error frame threshold: {configs['pass_frame_thresh']}, error frame threshold within {configs['relaxed_range']}m: {configs['pass_frame_thresh_with_relaxed_range']}")
    

    # Overwrite config batch folder with one from the environment
    assert os.environ.get('BATCH_NAME') is not None, f"Batch name is empty in environment. Please use this instead of config."
    configs['batch_folder'] = [os.environ['BATCH_NAME']]
    print(f"Replaced config file `batch_folders` with: {configs['batch_folder']}")


    # Download data from s3
    s3_to_hpc_json_path = s3_to_local_download(
        batch_name=configs["batch_folder"][0],
        destination_dir=configs["root_path"],
        config_path=args.config)



    # load data
    # data = utils.load_data(configs, False)
    batch_confirmed = utils.get_batch_folders(configs)
    # data = utils.load_data(configs)

    with open(results_file, "a") as f:
        f.write(f"Evaluate batch: {batch_confirmed}\n")
        f.write(f"Distance intervals for iou checking: {configs['distance']}\n")
        f.write(f"IoU threshold for RNC_C: {[float(f'{iou_t - 0.05:.2f}') for iou_t in configs['iou_threshold']]}, IoU threshold for other cams: {configs['iou_threshold']}\n")
        f.write(f"Error frame threshold: {configs['pass_frame_thresh']}, error frame threshold within {configs['relaxed_range']}m: {configs['pass_frame_thresh_with_relaxed_range']}\n")

    total_session = 0
    total_frames = 0
    passed_session = 0
    passed_frames = 0
    passed_session_relax = [0 for _ in range(len(configs['relaxed_range']))]
    passed_frames_relax = [0 for _ in range(len(configs['relaxed_range']))]

    for batch in batch_confirmed:
        batch_path = osp.join(configs["root_path"], batch)
        
        batch_data = utils.get_data(batch_path, configs['cams'], True)
        trunc_2d_cnt = {"-1":0, "10":0, "20":0, "30":0, "40":0}
        occ_2d_cnt = {"-1":0, "10":0, "20":0, "30":0, "40":0}
        trunc_3d_cnt = {"-1":0, "10":0, "20":0, "30":0, "40":0}
        occ_3d_cnt = {"-1":0, "10":0, "20":0, "30":0, "40":0}
        trunc_occ_csv = osp.join(result_folder, batch, "trunc_occ_warning.csv")

    
    # for batch in data.keys():
        sessions = list(batch_data.keys())
        sessions.sort()
        for session in sessions:
            total_session += 1
            save_folder = osp.join(result_folder, batch, batch+"_"+session)
            os.makedirs(save_folder, exist_ok=True)
            if configs["save_image"]:
                # create folder for each batch/session to save images
                
                
                # remove old images
                for file in os.listdir(save_folder):
                    if not (".png" in file):
                        continue
                    file_path = osp.join(save_folder, file)
                    if osp.isfile(file_path):
                        os.remove(file_path)
                trunc_folder = osp.join(save_folder, "trunc")
                occ_folder = osp.join(save_folder, "occ")
                iou_folder = osp.join(save_folder, "iou")
                os.makedirs(trunc_folder, exist_ok=True)
                os.makedirs(occ_folder, exist_ok=True)
                os.makedirs(iou_folder, exist_ok=True)
                for cur_folder in [iou_folder, trunc_folder, occ_folder]:
                    for file in os.listdir(cur_folder):
                        if not (".png" in file):
                            continue
                        file_path = osp.join(cur_folder, file)
                        if osp.isfile(file_path):
                            os.remove(file_path)

            if configs["log_session"]:
                log_file = osp.join(save_folder, configs["log_file"])
                if osp.isfile(log_file):
                    os.remove(log_file)
                for handler in logger.handlers[:]:  # Iterate over a copy of the list
                    if isinstance(handler, logging.FileHandler):
                        handler.close()  # Close the old handler
                        logger.removeHandler(handler)  # Remove the old handler
                log_handler = logging.FileHandler(log_file)
                log_handler.setFormatter(formatter)
                logger.addHandler(log_handler)
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Sanity check log - {current_time}")
                logger.info(f"Distance intervals for iou checking: {configs['distance']}")
                logger.info(f"IoU threshold for RNC_C: {[float(f'{iou_t - 0.05:.2f}') for iou_t in configs['iou_threshold']]}, IoU threshold for other cams: {configs['iou_threshold']}")
                logger.info(f"Error frame threshold: {configs['pass_frame_thresh']}, error frame threshold within {configs['relaxed_range']}m: {configs['pass_frame_thresh_with_relaxed_range']}")
            
            # associate_set = set()
            shape_id = 0
            # check image shape
            if "FNC" in configs["cams"] or "FWC_C" in configs["cams"]:
                cam_dirs = os.listdir(batch_data[session]["image_path"])
                cam_dir = [cam_folder for cam_folder in cam_dirs if utils.camera_map["FNC"] in cam_folder][0]
                image_folder = osp.join(batch_data[session]["image_path"], cam_dir, "png_files")
                image_files = os.listdir(image_folder)
                image_files = [i for i in image_files if ".png" in i]
                image_path = osp.join(image_folder, image_files[0])
                image = cv2.imread(image_path)
                cur_image_shape = image.shape
                if cur_image_shape[0] == image_shape_1["FNC"][0] and cur_image_shape[1] == image_shape_1["FNC"][1]:
                    shape_id = 0
                elif cur_image_shape[0] == image_shape_2["FNC"][0] and cur_image_shape[1] == image_shape_2["FNC"][1]:
                    shape_id = 1
                else:
                    logger.warning(f"unexpected image shape: {cur_image_shape}")

            session_image_shape = image_shape[shape_id]

            error_frame_cnt = set()
            error_frame_cnt_relax = [set() for _ in range(len(configs['relaxed_range']))]
            logger.critical(f"Processing batch: {batch}, Session: {session}")
            with open(results_file, "a") as f:
                f.write(f"\nEvaluate batch: {batch}, session: {session}\n")
            plot_id = 0
            trunc_id = 0
            occ_id = 0
            error_cnt = 0
            error_cnt_relax = [0 for _ in range(len(configs['relaxed_range']))]
            warning_cnt = 0
            warning_cnt_relax = [0 for _ in range(len(configs['relaxed_range']))]
            labels_2d = batch_data[session]["labels"]["2d"]
            labels_3d = batch_data[session]["labels"]["3d"]
            calibs = batch_data[session]["calibs"]
            image_files = batch_data[session]["images"]

            matching_dict = dict()
            # sanity check 3d labels
            for frame in range(len(labels_3d)):
                matching_dict[frame] = dict()
                label_frame_3d = labels_3d[str(frame)]
                for label in label_frame_3d:
                    if "parking" in label[3] or "dont" in label[3]:
                        continue
                    # associate_set.add((label[1], label[2]))
                    matching_dict[frame][int(label[2])] = dict()
                    matching_dict[frame][int(label[2])]["occ"] = int(label[5])
                    if label[4] not in ["10", "20", "30", "40"] or label[5] not in ["10", "20", "30", "40"]:
                        logger.error(f"unexpected 3D label: {label}")
                        error_cnt += 1
                        error_frame_cnt.add(frame)
                        for range_idx in range(len(configs['relaxed_range'])):
                            if np.sqrt(float(label[10])**2 + float(label[11])**2) < configs['relaxed_range'][range_idx]:
                                error_cnt_relax[range_idx] += 1
                                error_frame_cnt_relax[range_idx].add(frame)
                    if label[4] in ["-1", "10", "20", "30", "40"] and label[5] in ["-1", "10", "20", "30", "40"]:
                        trunc_3d_cnt[label[4]] += 1
                        occ_3d_cnt[label[5]] += 1

            num_frames = 0
            for cam in configs["cams"]:
                if num_frames == 0:
                    num_frames = len(labels_2d[cam])
                else:
                    assert num_frames == len(labels_2d[cam])

            frame_set = set(range(num_frames))
            total_frames += num_frames

            for cam in configs["cams"]:
                cur_image_shape = image_shape[shape_id][cam]
                image_cam = image_files[cam]["images"]
                image_timestamps = image_files[cam]["timestamps"]
                average_iou = []
                logger.critical(f"Camera: {cam}")
                intrinsic = calibs[cam]["intrinsic"]
                extrinsic = calibs[cam]["extrinsic"]
                for frame in range(len(labels_2d[cam])):
                    cam_label_frame_2d = labels_2d[cam][frame]
                    cam_label_frame_3d = labels_3d.get(str(frame), [])
                    #sanity check for trunc
                    if configs["check_trunc"]:
                        for idx, label in enumerate(cam_label_frame_2d):
                            if "parking" in label[0] or "dont" in label[0]:
                                continue
                            if label[1] not in ["10", "20", "30", "40"] or label[2] not in ["10", "20", "30", "40"]:
                                logger.error(f"unexpected 2D label frame id: {frame}, label: {label}")
                                error_cnt += 1
                                error_frame_cnt.add(frame)
                                for range_idx in range(len(configs['relaxed_range'])):
                                    if np.sqrt(float(label[11])**2 + float(label[13])**2) < configs['relaxed_range'][range_idx]:
                                        error_cnt_relax[range_idx] += 1
                                        error_frame_cnt_relax[range_idx].add(frame)
                            if label[1] in ["-1", "10", "20", "30", "40"] and label[2] in ["-1", "10", "20", "30", "40"]:
                                trunc_2d_cnt[label[1]] += 1
                                occ_2d_cnt[label[2]] += 1

                            # check truncation attr and box coord
                            if abs(float(label[4])) <= 20 or abs(float(label[5])) <= 20 or abs(float(label[6]) - cur_image_shape[1]) <= 20 or abs(float(label[7]) - cur_image_shape[0]) <= 20:
                                pass
                            else:
                                if label[1] != "10":
                                    cur_image_path = image_cam[frame]
                                    cur_image_timestamp = image_timestamps[frame]
                                    cur_image = cv2.imread(cur_image_path)
                                    box_2d = [int(float(label[4])), int(float(label[5])), int(float(label[6])), int(float(label[7]))]
                                    if (cur_image[box_2d[1]][box_2d[0]][0] == 0 and cur_image[box_2d[1]][box_2d[0]][1] == 0 and cur_image[box_2d[1]][box_2d[0]][2] == 0) or \
                                       (cur_image[box_2d[3]][box_2d[0]][0] == 0 and cur_image[box_2d[3]][box_2d[0]][1] == 0 and cur_image[box_2d[3]][box_2d[0]][2] == 0) or \
                                       (cur_image[box_2d[1]][box_2d[2]][0] == 0 and cur_image[box_2d[1]][box_2d[2]][1] == 0 and cur_image[box_2d[1]][box_2d[2]][2] == 0) or \
                                       (cur_image[box_2d[3]][box_2d[2]][0] == 0 and cur_image[box_2d[3]][box_2d[2]][1] == 0 and cur_image[box_2d[3]][box_2d[2]][2] == 0):
                                        pass
                                    else:
                                        if configs["save_image"]:
                                            add_txt = f"Ploted image id in trunc: {trunc_id} - "
                                        if configs["save_image"]:
                                            cv2.rectangle(cur_image, (int(float(label[4])), int(float(label[5]))), (int(float(label[6])), int(float(label[7]))), (0,0,255), 2)
                                            image_info_text = f"batch: {batch}, session: {session}, cam: {cam}, frame: {frame}, Image timestamp: {cur_image_timestamp}"
                                            obj_info_text = f"obj_id: {idx+1}, type: {label[0]}, trunc: {label[1]}, occ: {label[2]}, box: ({label[4]}, {label[5]}), ({label[6]}, {label[7]})"
                                            cv2.putText(cur_image, image_info_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                            cv2.putText(cur_image, obj_info_text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                            save_image_name = f"{trunc_id}_check_trunc_attr_{cam}_frame_{frame}.png"
                                            for range_idx in range(len(configs['relaxed_range'])):
                                                if np.sqrt(float(label[11])**2 + float(label[13])**2) < configs['relaxed_range'][range_idx]:
                                                    save_image_name = f"{trunc_id}_check_trunc_attr_{cam}_frame_{frame}_within_{configs['relaxed_range'][range_idx]}m.png"
                                                    break
                                            cv2.imwrite(osp.join(trunc_folder, save_image_name), cur_image)
                                            trunc_id += 1
                                        logger.warning(add_txt + f"Need to check 2D box truncation attr frame id: {frame}, label: {label}")
                                        with open(trunc_occ_csv, "a") as f:
                                            f.write(f"{batch},{session},trunc,{idx+1},{cam},{frame},{label[1]},{label[2]},{label[4]},{label[5]},{label[6]},{label[7]}\n")
                                        warning_cnt += 1
                                        for range_idx in range(len(configs['relaxed_range'])):
                                            if np.sqrt(float(label[11])**2 + float(label[13])**2) < configs['relaxed_range'][range_idx]:
                                                warning_cnt_relax[range_idx] += 1
                    
                    # convert labels from string to float
                    cur_label_2d = []
                    for i, cur_label in enumerate(cam_label_frame_2d):
                        # 0 object_id
                        # 1 object_type
                        # 2 truncated
                        # 3 occluded
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
                        cur_label_2d.append([i+1, cur_label[0], int(cur_label[1]), int(cur_label[2]), float(cur_label[4]), float(cur_label[5]), float(cur_label[6]), float(cur_label[7]), float(cur_label[8]), float(cur_label[9]), float(cur_label[10]), float(cur_label[11]), float(cur_label[12]), float(cur_label[13])])


                    cur_label_3d = []
                    for cur_label in cam_label_frame_3d:
                        # 0 track_id
                        # 1 object_type
                        # 2 length
                        # 3 width
                        # 4 height
                        # 5 location_x
                        # 6 location_y
                        # 7 location_z
                        # 8 rotation_z
                        cur_label_3d.append([int(cur_label[2]), cur_label[3], float(cur_label[7]), float(cur_label[8]), float(cur_label[9]), float(cur_label[10]), float(cur_label[11]), float(cur_label[12]), float(cur_label[13])])
                    
                    # match 2d and 3d labels
                    matching = [[] for _ in range(len(cur_label_2d))]
                    for i in range(len(cur_label_2d)):
                        for j in range(len(cur_label_3d)):
                            # check type, and 3d box size
                            if cur_label_2d[i][1] == cur_label_3d[j][1]:
                                if cur_label_2d[i][10] == cur_label_3d[j][2] and cur_label_2d[i][9] == cur_label_3d[j][3] and cur_label_2d[i][8] == cur_label_3d[j][4]:
                                    matching[i].append(j)
                    
                    # sanity check for projection
                    for i in range(len(cur_label_2d)):
                        if len(matching[i]) == 0:
                            logger.error(f"no matching 3D label found for 2D label frame id: {frame}, label: {cam_label_frame_2d[i]}")
                            error_cnt += 1
                            error_frame_cnt.add(frame)
                            for range_idx in range(len(configs['relaxed_range'])):
                                if np.sqrt(float(cur_label_2d[i][11])**2 + float(cur_label_2d[i][13])**2) < configs['relaxed_range'][range_idx]:
                                    error_cnt_relax[range_idx] += 1
                                    error_frame_cnt_relax[range_idx].add(frame)
                            continue
                        elif len(matching[i]) > 1:
                            matching_idx = -1
                            matching_dist = float("inf")
                            for j in matching[i]:
                                cur_matching_3d = cur_label_3d[j]
                                pos_3d = np.array([cur_matching_3d[5], cur_matching_3d[6], cur_matching_3d[7], 1.0]).reshape((4,1))
                                # project 3d box center to camera coord
                                pos_cam = np.matmul(extrinsic, pos_3d)[:3, 0]
                                cur_pos_diff = np.sqrt((pos_cam[0] - cur_label_2d[i][11])**2 + (pos_cam[1] - cur_label_2d[i][12])**2 + (pos_cam[2] - cur_label_2d[i][13])**2)
                                if cur_pos_diff < matching_dist:
                                    matching_dist = cur_pos_diff
                                    matching_idx = j
                            matching[i] = [matching_idx]
                        
                        matched_3d_label = cur_label_3d[matching[i][0]]
                        if "parking" in matched_3d_label[1] or "dont" in matched_3d_label[1]:
                            continue
                        if cam not in matching_dict[frame][matched_3d_label[0]]:
                            matching_dict[frame][matched_3d_label[0]][cam] = [i, cur_label_2d[i][3]]
                        else:
                            print(f"Warning: multiple 2D labels matched to the same 3D label, batch: {batch}, session: {session}, frame: {frame}, cam: {cam}, 3D label: {matched_3d_label}, previous 2D label: {cam_label_frame_2d[matching_dict[frame][matched_3d_label[0]][cam][0]]}, current 2D label: {cam_label_frame_2d[i]}")
                            logger.warning(f"multiple 2D labels matched to the same 3D label, batch: {batch}, session: {session}, frame: {frame}, cam: {cam}, 3D label: {matched_3d_label}, previous 2D label: {cam_label_frame_2d[matching_dict[frame][matched_3d_label[0]][cam][0]]}, current 2D label: {cam_label_frame_2d[i]}")

                        # only consider small/medium vehicle and occ/trunc 10
                        if not "small_vehicle" in cur_label_2d[i][1] and not "medium_vehicle" in cur_label_2d[i][1]:
                            continue
                        if cur_label_2d[i][3] > configs["occ_filter"]:
                            continue
                        if cur_label_2d[i][2] > configs["trunc_filter"]:
                            # if abs(cur_label_2d[i][4]) <= 20 or abs(cur_label_2d[i][5]) <= 20 or abs(cur_label_2d[i][6] - session_image_shape[cam][1]) <= 20 or abs(cur_label_2d[i][7] - session_image_shape[cam][0]) <= 20:
                            continue
                        # project 3d box center to camera coord
                        cur_matching_3d = cur_label_3d[matching[i][0]]
                        # associate_set.discard((f'{frame}', f'{cur_matching_3d[0]}'))
                        cur_3d_box = [cur_matching_3d[2], cur_matching_3d[3], cur_matching_3d[4], cur_matching_3d[5], cur_matching_3d[6], cur_matching_3d[7], cur_matching_3d[8]]
                        corners = utils.get_bbox_corners(cur_3d_box)
                        corners = np.matmul(extrinsic, corners)
                        corners = np.matmul(intrinsic, corners)

                        corners_valid = [False]*9
                        for k in range(9):
                            if corners[2, k] > 0:
                                corners_valid[k] = True
                                corners[:2, k] /= corners[2, k]

                        convex_corners = np.array([[corners[0,k], corners[1,k]] for k in range(9) if corners_valid[k]])
                        cur_convex = ConvexHull(convex_corners).vertices
                        

                        iou = utils.calculate_iou([cur_label_2d[i][4], cur_label_2d[i][5], cur_label_2d[i][6], cur_label_2d[i][7]], convex_corners[cur_convex], cur_image_shape)
                        obj_distance = np.sqrt(cur_matching_3d[5]**2 + cur_matching_3d[6]**2)
                        iou_threshold = configs["iou_threshold"][-1]
                        if cam == "FWC_C" and obj_distance > configs["FWC_C_distance"]:
                            continue
                        if obj_distance < configs["distance"][0]:
                            continue
                        if obj_distance >= configs["distance"][-1]:
                            continue
                        for dist_idx in range(1, len(configs["distance"])-1):
                            if obj_distance < configs["distance"][dist_idx]:
                                iou_threshold = configs["iou_threshold"][dist_idx-1]
                                break
                        
                        if cam == "RNC_C":
                            iou_threshold -= 0.05

                        if iou < iou_threshold:
                            if configs["save_image"]:
                                add_txt = f"Ploted image id in iou: {plot_id} - "
                            logger.error(add_txt + f"low IoU between 2D box and projected 3D box, frame id: {frame}, 2D label id: {cur_label_2d[i][0]}, 3D label id: {cur_matching_3d[0]}, IoU: {iou:.3f} object distance: {obj_distance:.2f}, threshold: {iou_threshold:.2f}, 2D label: {cam_label_frame_2d[i]}, 3D label: {cam_label_frame_3d[matching[i][0]]}")
                            # get image and draw box
                            if configs["save_image"]:
                                cur_image_path = image_cam[frame]
                                cur_image_timestamp = image_timestamps[frame]
                                cur_image = cv2.imread(cur_image_path)
                                cv2.rectangle(cur_image, (int(float(cur_label_2d[i][4])), int(float(cur_label_2d[i][5]))), (int(float(cur_label_2d[i][6])), int(float(cur_label_2d[i][7]))), (0,0,255), 2)
                                utils.draw_bbox(cur_image, corners.T.astype(int), corners_valid)
                                orig_3d_label = cam_label_frame_3d[matching[i][0]]
                                iou_text = f"batch: {batch}, session: {session}, cam: {cam}, frame: {frame}, Image timestamp: {cur_image_timestamp}, 3D timestamp: {int(orig_3d_label[0])%1000000 / 1000}, IoU: {iou:.3f}, object distance: {obj_distance:.2f}"
                                cv2.putText(cur_image, iou_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                                label_2d_text = f"2D: obj_id: {cur_label_2d[i][0]}, type: {cur_label_2d[i][1]}, trunc: {cur_label_2d[i][2]}, occ: {cur_label_2d[i][3]}, l: {cur_label_2d[i][10]}, w: {cur_label_2d[i][9]}, h: {cur_label_2d[i][8]}"
                                cv2.putText(cur_image, label_2d_text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                label_3d_text = f"3D: trk_id: {orig_3d_label[2]}, type: {orig_3d_label[3]}, trunc: {orig_3d_label[4]}, occ: {orig_3d_label[5]}, l: {orig_3d_label[7]}, w: {orig_3d_label[8]}, h: {orig_3d_label[9]}"
                                cv2.putText(cur_image, label_3d_text, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                save_image_name = f"{plot_id}_{cam}_low_iou_{iou:.3f}_frame_{frame}.png"
                                for range_idx in range(len(configs['relaxed_range'])):
                                    if obj_distance < configs['relaxed_range'][range_idx]:
                                        save_image_name = f"{plot_id}_{cam}_low_iou_{iou:.3f}_frame_{frame}_within_{configs['relaxed_range'][range_idx]}m.png"
                                        break
                                cv2.imwrite(osp.join(iou_folder, save_image_name), cur_image)
                                plot_id += 1

                            error_cnt += 1
                            error_frame_cnt.add(frame)
                            for range_idx in range(len(configs['relaxed_range'])):
                                if obj_distance < configs['relaxed_range'][range_idx]:
                                    error_cnt_relax[range_idx] += 1
                                    error_frame_cnt_relax[range_idx].add(frame)
                        average_iou.append(iou)
                    
                    if configs["check_occ"]:
                        sorted_by_z = sorted(cur_label_2d, key=lambda x: float(x[13]))
                        for id_2d in range(1, len(sorted_by_z)):
                            matched_3d = cur_label_3d[matching[sorted_by_z[id_2d][0]-1][0]]
                            if sorted_by_z[id_2d][3] <= 20:
                                continue
                            x_min, y_min = sorted_by_z[id_2d][4], sorted_by_z[id_2d][5]
                            x_max, y_max = sorted_by_z[id_2d][6], sorted_by_z[id_2d][7]
                            cur_box_poly = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                            orin_area = cur_box_poly.area
                            cur_area = orin_area
                            if orin_area <= 0:
                                continue
                            for id_2d_check in range(id_2d):
                                x_min_check, y_min_check = sorted_by_z[id_2d_check][4], sorted_by_z[id_2d_check][5]
                                x_max_check, y_max_check = sorted_by_z[id_2d_check][6], sorted_by_z[id_2d_check][7]
                                check_box_poly = Polygon([(x_min_check, y_min_check), (x_max_check, y_min_check), (x_max_check, y_max_check), (x_min_check, y_max_check)])
                                if not cur_box_poly.intersects(check_box_poly):
                                    continue
                                # subtract intersection area and get the new polygon
                                cur_box_poly = cur_box_poly.difference(check_box_poly)
                                cur_area = cur_box_poly.area
                                if cur_area/orin_area < 0.4:
                                    break
                            if configs["ignore_opposite_lane"]:
                                obj_heading = matched_3d[8]
                                if np.abs(obj_heading) > np.pi*2/3:
                                    continue
                            if cur_area/orin_area > 0.4:
                                add_txt = ""
                                if configs["save_image"]:
                                    add_txt = f"Ploted image id in occ: {occ_id} - "
                                logger.warning(add_txt + f"Check 2D box occ attr frame id: {frame}, label: {sorted_by_z[id_2d]}")
                                with open(trunc_occ_csv, "a") as f:
                                    # write: batch, session, occ, obj_id, cam, frame, trunc, occ, x1, y1, x2, y2
                                    f.write(f"{batch},{session},occ,{sorted_by_z[id_2d][0]},{cam},{frame},{sorted_by_z[id_2d][2]},{sorted_by_z[id_2d][3]},{int(sorted_by_z[id_2d][4])},{int(sorted_by_z[id_2d][5])},{int(sorted_by_z[id_2d][6])},{int(sorted_by_z[id_2d][7])}\n")
                                # get image and draw box
                                if configs["save_image"]:
                                    cur_image_path = image_cam[frame]
                                    cur_image_timestamp = image_timestamps[frame]
                                    cur_image = cv2.imread(cur_image_path)
                                    cv2.rectangle(cur_image, (int(float(sorted_by_z[id_2d][4])), int(float(sorted_by_z[id_2d][5]))), (int(float(sorted_by_z[id_2d][6])), int(float(sorted_by_z[id_2d][7]))), (0,0,255), 2)
                                    image_info_text = f"batch: {batch}, session: {session}, cam: {cam}, frame: {frame}, Image timestamp: {cur_image_timestamp}"
                                    obj_info_text = f"obj_id: {sorted_by_z[id_2d][0]}, type: {sorted_by_z[id_2d][1]}, trunc: {sorted_by_z[id_2d][2]}, occ: {sorted_by_z[id_2d][3]}, box: ({sorted_by_z[id_2d][4]}, {sorted_by_z[id_2d][5]}), ({sorted_by_z[id_2d][6]}, {sorted_by_z[id_2d][7]})"
                                    cv2.putText(cur_image, image_info_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                    cv2.putText(cur_image, obj_info_text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                    save_image_name = f"{occ_id}_check_occ_attr_{cam}_frame_{frame}.png"
                                    for range_idx in range(len(configs['relaxed_range'])):
                                        if np.sqrt(float(sorted_by_z[id_2d][11])**2 + float(sorted_by_z[id_2d][13])**2) < configs['relaxed_range'][range_idx]:
                                            save_image_name = f"{occ_id}_check_occ_attr_{cam}_frame_{frame}_within_{configs['relaxed_range'][range_idx]}m.png"
                                            break
                                    cv2.imwrite(osp.join(occ_folder, save_image_name), cur_image)
                                    occ_id += 1

                                warning_cnt += 1
                                for range_idx in range(len(configs['relaxed_range'])):
                                    if np.sqrt(float(sorted_by_z[id_2d][11])**2 + float(sorted_by_z[id_2d][13])**2) < configs['relaxed_range'][range_idx]:
                                        warning_cnt_relax[range_idx] += 1
                
                if len(average_iou) > 0:
                    logger.critical(f"Average IoU for camera {cam}: {np.mean(average_iou):.2f}")
                    with open(results_file, "a") as f:
                        f.write(f"Average IoU for camera {cam}: {np.mean(average_iou):.2f}\n")
                
            writing_csv_file = osp.join(save_folder, f"association.csv")
            with open(writing_csv_file, 'a') as csvfile:
                csvfile.write(f"frame_id, track_id, object_type, associated_cams, object_id, 3d_occ, iou, dist\n")

            for frame in matching_dict.keys():
                for track_id in matching_dict[frame].keys():
                    matching_label_3d = matching_dict[frame][track_id]
                    frame_3d_label = labels_3d.get(str(frame), [])
                    track_label_3d = [label for label in frame_3d_label if f'{label[2]}' == str(track_id)]
                    assert len(track_label_3d) == 1
                    cur_3d_label = track_label_3d[0]
                    cur_3d_box = [float(cur_3d_label[7]), float(cur_3d_label[8]), float(cur_3d_label[9]), float(cur_3d_label[10]), float(cur_3d_label[11]), float(cur_3d_label[12]), float(cur_3d_label[13])]
                    corners = utils.get_bbox_corners(cur_3d_box)
                    
                    with open(writing_csv_file, 'a') as csvfile:
                        for key in matching_dict[frame][track_id].keys():
                            if key in configs["cams"]:
                                extrinsic = calibs[key]["extrinsic"]
                                intrinsic = calibs[key]["intrinsic"]
                                cur_corners = np.matmul(extrinsic, corners)
                                cur_corners = np.matmul(intrinsic, cur_corners)

                                corners_valid = [False]*9
                                for k in range(9):
                                    if cur_corners[2, k] > 0:
                                        corners_valid[k] = True
                                        cur_corners[:2, k] /= cur_corners[2, k]

                                convex_corners = np.array([[cur_corners[0,k], cur_corners[1,k]] for k in range(9) if corners_valid[k]])
                                if corners_valid.count(True) == 0:
                                    iou = 0.0
                                else:
                                    cur_convex = ConvexHull(convex_corners).vertices
                                    matched_label_2d = labels_2d[key][frame][matching_dict[frame][track_id][key][0]]
                                    iou = utils.calculate_iou([float(matched_label_2d[4]), float(matched_label_2d[5]), float(matched_label_2d[6]), float(matched_label_2d[7])], convex_corners[cur_convex], session_image_shape[key])
                                obj_distance = np.sqrt(float(cur_3d_label[10])**2 + float(cur_3d_label[11])**2)
                                csvfile.write(f"{frame}, {track_id}, {cur_3d_label[3]}, {key}, {matching_dict[frame][track_id][key][0]+1}, {cur_3d_label[5]}, {iou:.3f}, {obj_distance:.2f}\n")

                    if len(matching_label_3d) == 1:
                        if matching_label_3d["occ"] != 40:
                            logger.error(f"3D label not associated to any 2D label, frame id: {frame}, label: {track_label_3d[0]}")
                            error_cnt += 1
                            error_frame_cnt.add(frame)
                            for range_idx in range(len(configs['relaxed_range'])):
                                if np.sqrt(float(track_label_3d[0][10])**2 + float(track_label_3d[0][11])**2) < configs['relaxed_range'][range_idx]:
                                    error_cnt_relax[range_idx] += 1
                                    error_frame_cnt_relax[range_idx].add(frame)
                    else:
                        occ_flag = True
                        for cam in configs["cams"]:
                            if cam not in matching_label_3d:
                                continue
                            if matching_label_3d[cam][1] < matching_label_3d["occ"]:
                                logger.error(f"3D label occ attribute is smaller than associated 2D label, frame id: {frame}, label: {track_label_3d[0]}, associated 2D label: {cam}, 2D object id: {matching_label_3d[cam][0]}")
                                error_cnt += 1
                                error_frame_cnt.add(frame)
                                for range_idx in range(len(configs['relaxed_range'])):
                                    if np.sqrt(float(track_label_3d[0][10])**2 + float(track_label_3d[0][11])**2) < configs['relaxed_range'][range_idx]:
                                        error_cnt_relax[range_idx] += 1
                                        error_frame_cnt_relax[range_idx].add(frame)
                                occ_flag = False
                            # if any associated 2D label occ is the same as 3D label occ
                            if matching_label_3d[cam][1] == matching_label_3d["occ"]:
                                occ_flag = False
                        if occ_flag:
                            cam_obj_id = [[cam, matching_label_3d[cam][0], matching_label_3d[cam][1]] for cam in matching_label_3d.keys() if cam in configs["cams"]]
                            logger.error(f"Check 3D label occ attribute, frame id: {frame}, label: {track_label_3d[0]}, associated 2D info: {cam_obj_id}")
                            error_cnt += 1
                            error_frame_cnt.add(frame)
                            for range_idx in range(len(configs['relaxed_range'])):
                                if np.sqrt(float(track_label_3d[0][10])**2 + float(track_label_3d[0][11])**2) < configs['relaxed_range'][range_idx]:
                                    error_cnt_relax[range_idx] += 1
                                    error_frame_cnt_relax[range_idx].add(frame)

            # if len(associate_set) > 0:
            #     for item in associate_set:
            #         frame_id, track_id = item
            #         frame_3d_label = labels_3d.get(frame_id, [])
            #         track_label_3d = [label for label in frame_3d_label if f'{label[2]}' == track_id]
            #         assert len(track_label_3d) <= 1
            #         if track_label_3d[0][3] not in ["small_vehicle", "medium_vehicle"]:
            #             continue
            #         if track_label_3d[0][5] in ["30", "40"]:
            #             continue
            #         logger.error(f"3D label not associated to any 2D label, frame id: {frame_id}, label: {track_label_3d}")
            #         print(f"3D label not associated to any 2D label label: {track_label_3d}")
            #         error_cnt += 1
            #         error_frame_cnt.add(int(frame_id))
            #         for range_idx in range(len(configs['relaxed_range'])):
            #             if np.sqrt(float(track_label_3d[0][10])**2 + float(track_label_3d[0][11])**2) < configs['relaxed_range'][range_idx]:
            #                 error_cnt_relax[range_idx] += 1
            #                 error_frame_cnt_relax[range_idx].add(int(frame_id))
            
            occ_2d_total = sum(occ_2d_cnt.values())
            trunc_2d_total = sum(trunc_2d_cnt.values())
            occ_3d_total = sum(occ_3d_cnt.values())
            trunc_3d_total = sum(trunc_3d_cnt.values())
            
            trunc_2d_percentage = {k: f'{v/trunc_2d_total*100:.2f}%' for k,v in trunc_2d_cnt.items()}
            trunc_3d_percentage = {k: f'{v/trunc_3d_total*100:.2f}%' for k,v in trunc_3d_cnt.items()}
            occ_2d_percentage = {k: f'{v/occ_2d_total*100:.2f}%' for k,v in occ_2d_cnt.items()}
            occ_3d_percentage = {k: f'{v/occ_3d_total*100:.2f}%' for k,v in occ_3d_cnt.items()}
            
            logger.critical(f"Truncation attr 2D: {trunc_2d_cnt}, 3D: {trunc_3d_cnt}")
            logger.critical(f"Truncation attr percentage 2D: {trunc_2d_percentage}, 3D: {trunc_3d_percentage}")
            logger.critical(f"Occlusion attr 2D: {occ_2d_cnt}, 3D: {occ_3d_cnt}")
            logger.critical(f"Occlusion attr percentage 2D: {occ_2d_percentage}, 3D: {occ_3d_percentage}")
            
            print(f"Truncation attr 2D: {trunc_2d_cnt}, 3D: {trunc_3d_cnt}")
            print(f"Truncation attr percentage 2D: {trunc_2d_percentage}, 3D: {trunc_3d_percentage}")
            print(f"Occlusion attr 2D: {occ_2d_cnt}, 3D: {occ_3d_cnt}")
            print(f"Occlusion attr percentage 2D: {occ_2d_percentage}, 3D: {occ_3d_percentage}")

            accepted_frames = frame_set.difference(error_frame_cnt)
            accepted_frames_relax = []
            for range_idx in range(len(configs['relaxed_range'])):
                accepted_frames_relax.append(frame_set.difference(error_frame_cnt_relax[range_idx]))
                passed_frames_relax[range_idx] += len(accepted_frames_relax[range_idx])
            passed_frames += len(accepted_frames)
            

            
            
            logger.critical(f"In batch: {batch}, session: {session}, total error: {error_cnt}, total warning: {warning_cnt}, total frames: {num_frames}, error frames: {len(error_frame_cnt)}, accepted frames: {len(accepted_frames)}, accept rate: {len(accepted_frames)/num_frames*100:.2f}%")
            logger.critical(f"Error frames: {sorted(list(error_frame_cnt))}")
            logger.critical(f"Passed frames: {sorted(list(accepted_frames))}")
            print(f"In batch: {batch}, session: {session}, total error: {error_cnt}, total warning: {warning_cnt}, total frames: {num_frames}, error frames: {len(error_frame_cnt)}, accepted frames: {len(accepted_frames)}, accept rate: {len(accepted_frames)/num_frames*100:.2f}%")
            for range_idx in range(len(configs['relaxed_range'])):
                logger.critical(f"In batch: {batch}, session: {session}, total error: {error_cnt_relax[range_idx]}, total warning: {warning_cnt_relax[range_idx]}, total frames: {num_frames}, error frames: {len(error_frame_cnt_relax[range_idx])} within {configs['relaxed_range'][range_idx]}m, accepted frames: {len(accepted_frames_relax[range_idx])}, accept rate: {len(accepted_frames_relax[range_idx])/num_frames*100:.2f}%")
                logger.critical(f"Error frames within {configs['relaxed_range'][range_idx]}m: {sorted(list(error_frame_cnt_relax[range_idx]))}")
                logger.critical(f"Passed frames within {configs['relaxed_range'][range_idx]}m: {sorted(list(accepted_frames_relax[range_idx]))}")
                print(f"In batch: {batch}, session: {session}, total error: {error_cnt_relax[range_idx]}, total warning: {warning_cnt_relax[range_idx]}, total frames: {num_frames}, error frames: {len(error_frame_cnt_relax[range_idx])} within {configs['relaxed_range'][range_idx]}m, accepted frames: {len(accepted_frames_relax[range_idx])}, accept rate: {len(accepted_frames_relax[range_idx])/num_frames*100:.2f}%")
            with open(results_file, "a") as f:
                f.write(f"Truncation attr 2D: {trunc_2d_cnt}, 3D: {trunc_3d_cnt}")
                f.write(f"\nTruncation attr percentage 2D: {trunc_2d_percentage}, 3D: {trunc_3d_percentage}\n")
                f.write(f"Occlusion attr 2D: {occ_2d_cnt}, 3D: {occ_3d_cnt}\n")
                f.write(f"Occlusion attr percentage 2D: {occ_2d_percentage}, 3D: {occ_3d_percentage}\n")
                f.write(f"Total error: {error_cnt}, total warning: {warning_cnt}, total frames: {num_frames}, error frames: {len(error_frame_cnt)}, accepted frames: {len(accepted_frames)}, accept rate: {len(accepted_frames)/num_frames*100:.2f}%\n")
                f.write(f"Error frames: {sorted(list(error_frame_cnt))}\n")
                f.write(f"Passed frames: {sorted(list(accepted_frames))}\n")
                for range_idx in range(len(configs['relaxed_range'])):
                    f.write(f"Total error: {error_cnt_relax[range_idx]}, total warning: {warning_cnt_relax[range_idx]}, total frames: {num_frames}, error frames: {len(error_frame_cnt_relax[range_idx])} within {configs['relaxed_range'][range_idx]}m, accepted frames: {len(accepted_frames_relax[range_idx])}, accept rate: {len(accepted_frames_relax[range_idx])/num_frames*100:.2f}%\n")
                    f.write(f"Error frames within {configs['relaxed_range'][range_idx]}m: {sorted(list(error_frame_cnt_relax[range_idx]))}\n")
                    f.write(f"Passed frames within {configs['relaxed_range'][range_idx]}m: {sorted(list(accepted_frames_relax[range_idx]))}\n")
            if len(error_frame_cnt) <= configs["pass_frame_thresh"]:
                with open(results_file, "a") as f:
                    f.write(f"Batch {batch}, session {session} passed the sanity check!\n")
                passed_session += 1
            for range_idx in range(len(configs['relaxed_range'])):
                if len(error_frame_cnt_relax[range_idx]) <= configs["pass_frame_thresh_with_relaxed_range"]:
                    with open(results_file, "a") as f:
                        f.write(f"Batch {batch}, session {session} within {configs['relaxed_range'][range_idx]}m passed the sanity check!\n")
                    passed_session_relax[range_idx] += 1
            
    
    with open(results_file, "a") as f:
        f.write(f"\nTotal sessions: {total_session}, passed sessions: {passed_session}\n")
        f.write(f"Total frames: {total_frames}, passed frames: {passed_frames}\n")
        for range_idx in range(len(configs['relaxed_range'])):
            f.write(f"Total sessions: {total_session}, passed sessions: {passed_session_relax[range_idx]} within {configs['relaxed_range'][range_idx]}m\n")
            f.write(f"Total frames: {total_frames}, passed frames: {passed_frames_relax[range_idx]} within {configs['relaxed_range'][range_idx]}m\n")
        # f.write(f"Total sessions: {total_session}, passed sessions: {passed_session_relax} within {configs['relaxed_range']}m\n")
        # f.write(f"Total frames: {total_frames}, passed frames: {passed_frames_relax} within {configs['relaxed_range']}m\n")
        f.write("Sanity check finished!\n")
        f.write("========================================\n")
    
    print(f"Total sessions: {total_session}, passed sessions: {passed_session}")
    print(f"Total frames: {total_frames}, passed frames: {passed_frames}")
    for range_idx in range(len(configs['relaxed_range'])):
        print(f"Total sessions: {total_session}, passed sessions: {passed_session_relax[range_idx]} within {configs['relaxed_range'][range_idx]}m")
        print(f"Total frames: {total_frames}, passed frames: {passed_frames_relax[range_idx]} within {configs['relaxed_range'][range_idx]}m")
    # print(f"Total sessions: {total_session}, passed sessions: {passed_session_relax} within {configs['relaxed_range']}m")
    
    # print(f"Total frames: {total_frames}, passed frames: {passed_frames_relax} within {configs['relaxed_range']}m")

    # Upload the results of datascreen to S3          
    uploaded_s3dir = upload_to_s3(
        config_path=args.config,
        batch_name=configs["batch_folder"][0],
        src_dir=osp.join(result_folder, batch),
        standalone_files=[results_file]
    )
    print('Uploaded results of datascreen to S3 at:', uploaded_s3dir)

    passing_sessions: dict = get_batch_passing_sessions(
        batch_summary_log_path=results_file,
        passing_distance_threshold=100
    )
    print(f'Found {len(passing_sessions.get(batch, []))} accepted sessions: {passing_sessions.get(batch, [])}')
    
    # Upload the sanity check files
    add_sanity_check_logs_to_s3_to_hpc_json(
        sess_sanity_check_dir=osp.join(result_folder, batch),
        s3_session_sanity_check_dir=uploaded_s3dir,
        config_path=args.config, 
        s3_to_hpc_json_path=s3_to_hpc_json_path,
        dest_path=os.path.join(result_folder, "tmp_s3_to_hpc.json"))
    get_filtered_s3_to_hpc_json(
        config_path=args.config, 
        s3_to_hpc_json_path=os.path.join(result_folder, "tmp_s3_to_hpc.json"),
        dest_path=s3_to_hpc_json_path.replace(configs["s3_to_hpc_json"], "s3_to_hpc.json"), # Final path expected
        include=passing_sessions.get(batch, [])
    )

    # Write output to Airflow xcom if necessary
    data = {
        "dataset_s3path": s3_to_hpc_json_path,
        "datascreen_s3path": uploaded_s3dir,
        "passing": passing_sessions,
        # passing_sessions can theoretically contain more than 1 batch, although it should not happen in this script
        "num_passing": sum([len(ps) for ps in passing_sessions.values()])
    }
    write_xcom_file(data)

    # Delete results of datascreen from local machine
    delete_local_dir(result_folder)

    # Delete downloaded batch data (images, anno files, etc) from local machine
    downloaded_batch_data_dir = osp.join(configs["root_path"], batch)
    delete_local_dir(downloaded_batch_data_dir)

if __name__ == "__main__":
    main()