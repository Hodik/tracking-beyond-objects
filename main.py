import cv2
import mmcv
import torch
import numpy as np
from ultralytics import YOLO
from action_detection import TaskInfo, StdetPredictor
from object_tracker import ObjectTracker
from itertools import chain

from mmengine import Config

from display import plot


def get_person_cls(clss):
    return next(filter(lambda x: x == "person", clss), 0)


people: dict[int, ObjectTracker] = {}
objects: dict[int, ObjectTracker] = {}

# Load the YOLOv8 model
recognizer_config = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
recognizer_checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
label = "mmaction2/tools/data/kinetics/label_map_k400.txt"

# Open the video file
cameras = ["bowling.mp4"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.to(device)

cfg = 'mmaction2/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'
checkpoint = 'https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth'
action_config = Config.fromfile(cfg)
clip_vis_length = 8
predict_stepsize = 40
stdet_input_shortside = 256
score_thr = 0.4
label_map = 'mmaction2/tools/data/ava/label_map.txt'
stdet_predictor = StdetPredictor(action_config, checkpoint, device, score_thr, label_map)

caps = [cv2.VideoCapture(camera) for camera in cameras]

val_pipeline = action_config.val_pipeline
sampler = [x for x in val_pipeline
            if x['type'] == 'SampleAVAFrames'][0]
clip_len, frame_interval = sampler['clip_len'], sampler[
    'frame_interval']
window_size = clip_len * frame_interval
h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
stdet_input_size = mmcv.rescale_size(
    (w, h), (stdet_input_shortside, np.Inf))
img_norm_cfg = dict(
    mean=np.array(action_config.model.data_preprocessor.mean),
    std=np.array(action_config.model.data_preprocessor.std),
    to_rgb=False)
buffer_size = window_size - predict_stepsize
frame_start = window_size // 2 - (clip_len // 2) * frame_interval
frames_inds = [
    frame_start + frame_interval * i for i in range(clip_len)
]
display_size = (w, h)
ratio = tuple(n / o for n, o in zip(stdet_input_size, display_size))
read_id = 1

fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for MP4 format
fps = 25  # Frames per second
frame_size = (1280, 720)  # Desired output frame size (adjust based on your frames)

# Create a video writer object
video_writer = cv2.VideoWriter("output.mp4", fourcc, fps, frame_size)


def merge_action_buffers(action_buffers):
    return {i: action for i, action in chain(*action_buffers)}


def main():
    # Loop through the video frames
    
    read_id = 0
    actions_buffer = {}

    while all(cap.isOpened() for cap in caps):
        # Read a frame from the videoc
        tasks = [TaskInfo(clip_vis_length, frames_inds, ratio) for _ in caps]
        frames = [{"frames": [], "processed_frames": [], "keyframe": {"frame": None, "boxes": None}} for _ in caps]
        success = True

        while success and all(len(f['frames']) < window_size for f in frames):
            for cam_id, cap in enumerate(caps):
                success, frame = cap.read()
                if success:
                    results = model.track(frame, persist=True)
                    frames[cam_id]['frames'].append(mmcv.imresize(frame, display_size))
                    processed_frame = mmcv.imresize(
                        frame, stdet_input_size).astype(np.float32)
                    _ = mmcv.imnormalize_(processed_frame, **img_norm_cfg)
                    frames[cam_id]['processed_frames'].append(processed_frame)


                    if len(frames[cam_id]['frames']) == window_size // 2:
                        boxes = results[0].boxes.xyxy
                        clss = results[0].boxes.cls
                        person_class = get_person_cls(clss)

                        frames[cam_id]['keyframe']['frame'], frames[cam_id]['keyframe']['boxes'] = frame,  results[0].boxes[clss == person_class]

                        person_boxes = boxes[clss == person_class].to(device)
                        tasks[cam_id].add_bboxes(person_boxes)

                    annotated_frame = plot(results[0], actions=actions_buffer)
                    cv2.imshow(f"Camera {cam_id}", annotated_frame)
                    video_writer.write(annotated_frame)
                    # Pause until a key is pressed, using 0 as a timeout to check for key presses
                    key = cv2.waitKey(1)

                    # Handle key presses
                    if key == ord("q"):  # Quit
                        break
                    elif key == ord(" "):  # Pause/resume
                        while True:
                            key = cv2.waitKey(0) & 0xFF
                            if key == ord(" "):  # Unpause
                                break
                            elif key == ord("q"):  # Quit while paused
                                break
        
        preds = []
        for cam_task, cam_frames in zip(tasks, frames):
            cam_task.add_frames(read_id, frames, cam_frames['processed_frames'])
            stdet_predictor.predict(cam_task)
            if cam_task.action_preds and cam_frames['keyframe']['boxes'].id is not None:
                track_ids = cam_frames['keyframe']['boxes'].id.int().cpu().tolist()
                preds.append(zip(track_ids, cam_task.action_preds))
        actions_buffer = merge_action_buffers(preds)        
        read_id += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        map(lambda cap: cap.release(), caps)
        video_writer.release()
        cv2.destroyAllWindows()
        exit(0)
