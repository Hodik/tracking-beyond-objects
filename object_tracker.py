from collections import deque
import cv2
import numpy as np
from operator import itemgetter
import torch
from mmengine.dataset import pseudo_collate


class ObjectTracker:
    def __init__(self, _id, initial_box, cls_id, people, objects, frames_max_len):
        self._id = _id
        self.boxes = [initial_box]
        self.cls_id = cls_id
        self.related_objects: dict[int, int] = {}
        self.people = people
        self.objects = objects
        self.frames = deque(maxlen=frames_max_len)

    def add_box(self, box):
        self.boxes.append(box)

    def add_related_object(self, track_id: int):
        if track_id in self.related_objects:
            self.related_objects[track_id] += 1
        else:
            self.related_objects[track_id] = 1

    def destroy_related_object(self, track_id: int):
        if track_id in self.related_objects:
            del self.related_objects[track_id]

    def remove_related_object(self, track_id: int):
        if track_id in self.related_objects:
            self.related_objects[track_id] -= 1
            if self.related_objects[track_id] == 0:
                self.destroy_related_object(track_id)

    def clean_related_objects(self):
        for track_id in list(self.related_objects.keys()):
            if track_id not in self.objects and track_id not in self.people:
                self.destroy_related_object(track_id)

    def bbox(self):

        all_boxes = [self.boxes[-1]]
        for track_id in self.related_objects:
            if track_id in self.objects:
                all_boxes.append(self.objects[track_id].boxes[-1])
            else:
                try:
                    all_boxes.append(self.people[track_id].boxes[-1])
                except KeyError:
                    print("KeyError", track_id, self.people.keys(), self.objects.keys())
                    raise KeyboardInterrupt

        x_min, y_min = min(b[0] for b in all_boxes), min(b[1] for b in all_boxes)
        x_max, y_max = max(b[2] for b in all_boxes), max(b[3] for b in all_boxes)
        return (x_min, y_min, x_max, y_max)

    def add_frame(self, frame):
        # if self.frames:
        #     frame_shape = self.frames[0].shape
        #     frame = cv2.resize(frame, frame_shape)
        if not frame.size:
            return
        frame = cv2.resize(frame, (250, 250))
        self.frames.append(np.array(frame[:, :, ::-1]))

    def get_action(self, recognizer: dict):
        if len(self.frames) == recognizer["sample_length"]:
            cur_windows = list(np.array(self.frames))
            cur_data = recognizer["data"].copy()
            cur_data["img_shape"] = self.frames.popleft().shape[:2]
            cur_data["imgs"] = cur_windows
            cur_data = recognizer["pipeline"](cur_data)
            print("shape of data feeding", cur_data['inputs'].shape)
            cur_data = pseudo_collate([cur_data])
            print("shape of data feeding", cur_data['inputs'].shape)
            with torch.no_grad():
                results = recognizer["model"].test_step(cur_data)[0]
            pred_scores = results.pred_score.tolist()
            score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
            score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
            return [(recognizer["labels"][k[0]], k[1]) for k in score_sorted[:5]]
        return None

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, __value: object) -> bool:
        return self._id == __value._id

    def __str__(self) -> str:
        return f"ObjectTracker {self._id} class <{self.cls_id}>, related: {self.related_objects}"


class PersonTracker(ObjectTracker): ...
