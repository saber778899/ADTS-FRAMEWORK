import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import argparse
import torch
import cv2
import yaml
import imageio
from tqdm import tqdm
import os.path as osp
import numpy as np
import os
import platform
import shutil
import time
import sys
sys.path.insert(0, '/home/liuhaoan/ADTS/AstroDetNet')
from AstroDetNet.utils.torch_utils import select_device, time_sync
from AstroDetNet.utils.general import check_img_size, scale_coords, non_max_suppression, check_imshow
from AstroDetNet.utils.datasets import LoadImages, LoadStreams
from AstroDetNet.models.experimental import attempt_load

from AstroDetNet.val import post_process_batch

# AstroSORT related modules
from AstroTracNet.utils.parser import get_config
from AstroTracNet.astro_sort import AstroSort

colors_list = [
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255],
        [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127],
        [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0],
        [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
    ]  # 27 colors

# Map for storing ID to color mappings
id_color_map = {}

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def draw_boxes(img, bbox, identities=None, offset=(0, 0), line_thickness=2):
    global id_color_map, colors_list
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        # Get ID
        id = int(identities[i]) if identities is not None else 0
        
        # Assign a consistent color for each ID
        if id not in id_color_map:
            id_color_map[id] = colors_list[len(id_color_map) % len(colors_list)]
        
        # Use the mapped color
        color = id_color_map[id]
        
        # Draw box and ID
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video options
    parser.add_argument('-p', '--video-path', default='/home/liuhaoan/ADTS/AstroDetNet/test_imgs/test_video_23.mp4', help='path to video file')

    parser.add_argument('--data', type=str, default='/home/liuhaoan/ADTS/AstroDetNet/data/JointBP_HumanParts.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)  # 128*8
    parser.add_argument('--save-size', type=int, default=960)
    parser.add_argument('--weights', default='/home/liuhaoan/ADTS/AstroDetNet/runs/train/exp9/best_mMR_0328.pt')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.7, help='Matching IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    
    parser.add_argument('--start', type=int, default=0, help='start time (s)')
    parser.add_argument('--end', type=int, default=-1, help='end time (s), -1 for remainder of video')
    parser.add_argument('--color', type=int, nargs='+', default=[255, 255, 255], help='person bbox color')
    parser.add_argument('--thickness', type=int, default=6, help='thickness of orientation lines')
    parser.add_argument('--alpha', type=float, default=0.4, help='origin and plotted alpha')
    
    parser.add_argument('--display', action='store_true', help='display inference results')
    parser.add_argument('--fps-size', type=int, default=1)
    parser.add_argument('--gif', action='store_true', help='create gif')
    parser.add_argument('--gif-size', type=int, nargs='+', default=[480, 270])

    # AstroSORT related parametersL
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--astro-sort-weights', type=str, default='/home/liuhaoan/ADTS/AstroTracNet/astro_sort/astro/checkpoint/exp12_best/checkpoint/ckpt.t7', help='AstroSORT model weights')
    parser.add_argument('--config-astrosort', type=str, default='/home/liuhaoan/ADTS/AstroTracNet/configs/astro_sort.yaml', help='AstroSORT config file')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder for tracking results')
    
    # SimpleStableKalmanFilter parameters (updated to match new implementation)
    parser.add_argument('--max-history-length', type=int, default=60, help='Maximum length of track history')
    parser.add_argument('--velocity-weight', type=float, default=0.6, help='Weight for velocity consistency in motion prediction')
    parser.add_argument('--appearance-weight', type=float, default=0.7, help='Weight for appearance features in matching')
    parser.add_argument('--update-threshold', type=int, default=30, help='Frames to maintain ID during occlusion')

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.video_path, img_size=imgsz, stride=stride, auto=True)

    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Initialize AstroSORT with updated parameters
    cfg = get_config()
    cfg.merge_from_file(args.config_astrosort)
    astrosort = AstroSort(args.astro_sort_weights,
                        max_dist=cfg.ASTROSORT.MAX_DIST, 
                        min_confidence=cfg.ASTROSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.ASTROSORT.NMS_MAX_OVERLAP, 
                        max_iou_distance=cfg.ASTROSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.ASTROSORT.MAX_AGE, 
                        n_init=cfg.ASTROSORT.N_INIT, 
                        nn_budget=cfg.ASTROSORT.NN_BUDGET,
                        use_cuda=True, 
                        max_history_length=args.max_history_length,
                        velocity_weight=args.velocity_weight,
                        appearance_weight=args.appearance_weight,
                        time_since_update_threshold=args.update_threshold)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    data['conf_thres_part'] = args.conf_thres  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = args.iou_thres  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = args.match_iou  # whether a body-part in matched with one body bbox

    cap = dataset.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.end == -1:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - fps * args.start)
    else:
        n = int(fps * (args.end - args.start))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gif_frames = []
    out_path = '{}_{}'.format(osp.splitext(args.video_path)[0])
    print("fps:", fps, "\t total frames:", n, "\t out_path:", out_path)

    # Prepare text file for MOT tracking results
    txt_path = os.path.join(args.output, osp.basename(args.video_path).split('.')[0] + '.txt')
    if args.save_txt and os.path.exists(txt_path):
        os.remove(txt_path)

    write_video = not args.display and not args.gif
    if write_video:
        writer = cv2.VideoWriter(out_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, 
            (int(args.save_size*w/h), args.save_size))
            
    dataset = tqdm(dataset, desc='Running inference', total=n)
    t0 = time_sync()
    for frame_idx, (path, img, im0, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # Run model inference
        out_ori = model(img, augment=True, scales=args.scales)[0]
        body_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=[0], num_offsets=data['num_offsets'])
        part_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=list(range(1, 1 + data['num_offsets']//2)), num_offsets=data['num_offsets'])
        
        # Post-process body and part detections
        bboxes, points, scores, _, _, _ = post_process_batch(
            data, img, [], [[im0.shape[:2]]], body_dets, part_dets)
        
        # Set line thickness
        args.line_thick = max(im0.shape[:2]) // 600 + 8

        # Detected body boxes and part information
        detected_bboxes = []  # Store detected body boxes and scores
        body_masks = []  # Record which body boxes are kept
        
        # First draw bodies and body parts
        for ind, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            [x1, y1, x2, y2] = bbox
            keep_body = True
            
            # Process body part detections based on dataset type
            if data['dataset'] == "HumanParts":  # data['num_offsets'] is 12
                has_part = False
                for cls_ind in range(data['num_offsets']//2):
                    t_score, t_bbox = point[cls_ind][2], point[cls_ind][3:]  # body-part, bbox format [x1, y1, x2, y2]
                    if t_score != 0:
                        has_part = True
                
                # Skip tracking bodies without valid parts
                if not has_part:
                    keep_body = False
            
            # Record bodies to track
            if keep_body:
                detected_bboxes.append((bbox, score, point))  # Add point information
                body_masks.append(ind)
        
        # Prepare for AstroSORT tracking
        xywh_bboxs = []
        confs = []
        
        # Convert detection results to AstroSORT format
        for bbox, score, _ in detected_bboxes:
            [x1, y1, x2, y2] = bbox
            xywh_obj = [x1 + (x2-x1)/2, y1 + (y2-y1)/2, x2-x1, y2-y1]  # Convert to center coordinates and dimensions
            xywh_bboxs.append(xywh_obj)
            confs.append([score])  # Use original detection confidence
        
        # Save a copy of the original image
        im0_original = im0.copy()

        # Update AstroSORT tracker
        if len(xywh_bboxs) > 0:
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            
            # Update AstroSORT tracker
            outputs = astrosort.update(xywhs, confss, im0.copy())
            
            # If there are tracking results
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                
                # Map tracking IDs to detection boxes
                for i, (bb_det, score, point) in enumerate(detected_bboxes):
                    min_dist = float('inf')
                    matched_id = None
                    
                    # Find the closest tracking box to this detection
                    for j, (bb_track, id_track) in enumerate(zip(bbox_xyxy, identities)):
                        # Calculate center point distance
                        det_center_x = (bb_det[0] + bb_det[2]) / 2
                        det_center_y = (bb_det[1] + bb_det[3]) / 2
                        track_center_x = (bb_track[0] + bb_track[2]) / 2
                        track_center_y = (bb_track[1] + bb_track[3]) / 2
                        
                        dist = np.sqrt((det_center_x - track_center_x)**2 + (det_center_y - track_center_y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = id_track
                    
                    # If a matching ID is found, use its color
                    if matched_id is not None and min_dist < 50:  # Set a distance threshold
                        # Assign fixed color for ID (if not already assigned)
                        if matched_id not in id_color_map:
                            id_color_map[matched_id] = colors_list[int(matched_id) % len(colors_list)]
                        
                        color = id_color_map[matched_id]
                        
                        # Draw body box
                        [x1, y1, x2, y2] = bb_det
                        cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)
                        
                        # Draw body parts (using same color)
                        if data['dataset'] == "HumanParts":
                            for cls_ind in range(data['num_offsets']//2):
                                t_score, t_bbox = point[cls_ind][2], point[cls_ind][3:]
                                if t_score != 0:
                                    [px1, py1, px2, py2] = t_bbox
                                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), 
                                                 color, thickness=args.line_thick)
                
                # Draw tracking ID labels
                for i, (box, id) in enumerate(zip(bbox_xyxy, identities)):
                    x1, y1, x2, y2 = [int(i) for i in box]
                    
                    # Use fixed color for this ID
                    if id not in id_color_map:
                        id_color_map[id] = colors_list[int(id) % len(colors_list)]
                    
                    color = id_color_map[id]
                    
                    # Draw ID label
                    label = '{}{:d}'.format("", id)
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 7, 7)[0]
                    cv2.rectangle(
                        im0, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                    cv2.putText(im0, label, (x1, y1 +
                                           t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 7, [255, 255, 255], 7)
                
                # Save MOT format results
                if args.save_txt:
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                    for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                        bbox_top = tlwh_bbox[0]
                        bbox_left = tlwh_bbox[1]
                        bbox_w = tlwh_bbox[2]
                        bbox_h = tlwh_bbox[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                        bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # MOT format
            else:
                # If no tracking results, restore original image
                im0 = im0_original
        else:
            # If no detections, increment tracker ages
            astrosort.increment_ages()
            # Restore original image
            im0 = im0_original

        if frame_idx == 0:
            t = time_sync() - t0
        else:
            t = time_sync() - t1

        # Add FPS display
        if not args.gif and args.fps_size:
            cv2.putText(im0, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)
                
        # Handle output
        if args.gif:
            gif_img = cv2.cvtColor(cv2.resize(im0, dsize=tuple(args.gif_size)), cv2.COLOR_RGB2BGR)
            if args.fps_size:
                cv2.putText(gif_img, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                    cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)
            gif_frames.append(gif_img)
        elif write_video:
            im0 = cv2.resize(im0, dsize=(int(args.save_size*w/h), args.save_size))
            writer.write(im0)
        else:
            cv2.imshow('', im0)
            cv2.waitKey(1)

        t1 = time_sync()
        if frame_idx == n - 1:
            break

    cv2.destroyAllWindows()
    cap.release()
    if write_video:
        writer.release()

    if args.gif:
        print('Saving GIF...')
        with imageio.get_writer(out_path + '.gif', mode="I", fps=fps) as writer:
            for idx, frame in tqdm(enumerate(gif_frames)):
                writer.append_data(frame)

    if args.save_txt or write_video:
        if platform.system() == 'Darwin':  # MacOS
            os.system('open ' + out_path + '.mp4')