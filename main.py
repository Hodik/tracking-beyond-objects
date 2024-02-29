import cv2
import torch
import numpy as np
from ultralytics import YOLO
from connected_objects import distance_with_iou
from object_tracker import ObjectTracker, PersonTracker

from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction2.mmaction.apis import init_recognizer
from mmaction2.mmaction.utils import get_str_type


def get_person_cls(clss):
    return next(filter(lambda x: x == "person", clss), 0)


people: dict[int, ObjectTracker] = {}
objects: dict[int, ObjectTracker] = {}

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")
recognizer_config = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
recognizer_checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
label = "mmaction2/tools/data/kinetics/label_map_k400.txt"

# Open the video file
video_path = "bowling.mp4"
cap = cv2.VideoCapture(video_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_recognizer():
    cfg = Config.fromfile(recognizer_config)
    model = init_recognizer(cfg, recognizer_checkpoint, device=device)
    data = dict(img_shape=(250, 250), modality='RGB', label=-1)
    with open(label, 'r') as f:
        labels = [line.strip() for line in f]
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in get_str_type(step['type']):
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if get_str_type(step['type']) in [
            'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
            'PyAVDecode', 'RawFrameDecode'
        ]:
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    return {"cfg": cfg, "model": model, "data": data, "labels": labels, "sample_length": sample_length, "pipeline": test_pipeline}

recognizer = setup_recognizer()

def post_process(results, frame=None):
    boxes = results[0].boxes.xyxy.int().cpu().tolist()
    if len(boxes) == 0:
        return

    clss = results[0].boxes.cls.cpu().tolist()

    if results[0].boxes.id is None or not len(results[0].boxes.id):
        return

    track_ids = results[0].boxes.id.int().cpu().tolist()
    class_names = results[0].names
    person_class = get_person_cls(clss)

    for box, cls, i in zip(boxes, clss, track_ids):
        if cls != person_class:
            if i in objects:
                objects[i].add_box(box)
            else:
                objects[i] = ObjectTracker(i, box, cls, people, objects, frames_max_len=recognizer['sample_length'])

    for p_box, p_cls, p_i in zip(boxes, clss, track_ids):
        if p_cls == person_class:
            if p_i in people:
                people[p_i].add_box(p_box)
            else:
                people[p_i] = PersonTracker(p_i, p_box, p_cls, people, objects, frames_max_len=recognizer['sample_length'])

            # calculate distance between people and objects
            for box, cls, i in zip(boxes, clss, track_ids):
                distance = distance_with_iou(p_box, box)
                if distance < 100:
                    people[p_i].add_related_object(i)
                else:
                    people[p_i].remove_related_object(i)

    for o_id in list(objects.keys()):
        if o_id not in track_ids:
            del objects[o_id]

    for p_id in list(people.keys()):
        if p_id not in track_ids:
            del people[p_id]

    for p in people:
        people[p].clean_related_objects()


    for p in people:
        bbox = people[p].bbox()
        cropped_image = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        people[p].add_frame(cropped_image)
        top_actions = people[p].get_action(recognizer)
        if top_actions:
            print(f"got actions {top_actions} for person {p}")
        # if cropped_image.size:
        #     cv2.imshow("cropped object ready for caption", cropped_image)
        #     cv2.waitKey(0)

import mmcv
from mmengine import Config
from action_detection import TaskInfo, StdetPredictor
from display import plot


cfg = 'mmaction2/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'
checkpoint = 'https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth'
action_config = Config.fromfile(cfg)
clip_vis_length = 8
predict_stepsize = 40
stdet_input_shortside = 256
score_thr = 0.4
label_map = 'mmaction2/tools/data/ava/label_map.txt'
stdet_predictor = StdetPredictor(action_config, checkpoint, device, score_thr, label_map)


def main():
    # Loop through the video frames
    val_pipeline = action_config.val_pipeline
    sampler = [x for x in val_pipeline
                if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler[
        'frame_interval']
    window_size = clip_len * frame_interval
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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
    actions_buffer = {}

    while cap.isOpened():
        # Read a frame from the videoc

        task = TaskInfo()
        task.clip_vis_length = clip_vis_length
        task.frames_inds = frames_inds
        task.ratio = ratio

        # read buffer
        frames = []
        processed_frames = []
        success = True
        keyframe = {
            "frame": None,
            "boxes": None
        }

        while success and len(frames) < window_size:
            success, frame = cap.read()
            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)
                
                frames.append(mmcv.imresize(frame, display_size))
                processed_frame = mmcv.imresize(
                    frame, stdet_input_size).astype(np.float32)
                _ = mmcv.imnormalize_(processed_frame, **img_norm_cfg)
                processed_frames.append(processed_frame)


                if len(frames) == window_size // 2:
                    boxes = results[0].boxes.xyxy
                    clss = results[0].boxes.cls
                    person_class = get_person_cls(clss)

                    keyframe['frame'], keyframe['boxes'] = frame,  results[0].boxes[clss == person_class]

                    person_boxes = boxes[clss == person_class].to(device)
                    task.add_bboxes(person_boxes)

                annotated_frame = plot(results[0], actions=actions_buffer)

                cv2.imshow("YOLOv8 Tracking", annotated_frame)
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
        task.add_frames(read_id, frames, processed_frames)
        stdet_predictor.predict(task)
        if task.action_preds and keyframe['boxes'].id is not None:
            track_ids = keyframe['boxes'].id.int().cpu().tolist()
            actions_buffer = {i: action for i, action in zip(track_ids, task.action_preds)}
        read_id += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)
