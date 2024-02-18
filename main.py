import cv2
from ultralytics import YOLO
from connected_objects import distance_with_iou
from typing import Self


def get_person_cls(clss):
    return next(filter(lambda x: x == "person", clss), 0)


class ObjectTracker:
    def __init__(self, _id, initial_box, cls_id):
        self._id = _id
        self.boxes = [initial_box]
        self.cls_id = cls_id
        self.related_objects: dict[int, int] = {}

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
            if track_id not in objects and track_id not in people:
                self.destroy_related_object(track_id)

    def bbox(self):

        all_boxes = [self.boxes[-1]]
        for track_id in self.related_objects:
            if track_id in objects:
                all_boxes.append(objects[track_id].boxes[-1])
            else:
                try:
                    all_boxes.append(people[track_id].boxes[-1])
                except KeyError:
                    print("KeyError", track_id, people.keys(), objects.keys())
                    raise KeyboardInterrupt

        x_min, y_min = min(b[0] for b in all_boxes), min(b[1] for b in all_boxes)
        x_max, y_max = max(b[2] for b in all_boxes), max(b[3] for b in all_boxes)
        return (x_min, y_min, x_max, y_max)

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, __value: object) -> bool:
        return self._id == __value._id

    def __str__(self) -> str:
        return f"ObjectTracker {self._id} class <{self.cls_id}>, related: {self.related_objects}"


class PersonTracker(ObjectTracker): ...


people: dict[int, ObjectTracker] = {}
objects: dict[int, ObjectTracker] = {}

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "bowling.mp4"
cap = cv2.VideoCapture(video_path)


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
                objects[i] = ObjectTracker(i, box, cls)

    for p_box, p_cls, p_i in zip(boxes, clss, track_ids):
        if p_cls == person_class:
            if p_i in people:
                people[p_i].add_box(p_box)
            else:
                people[p_i] = PersonTracker(p_i, p_box, p_cls)

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
        cv2.imshow("cropped object ready for caption", cropped_image)
        cv2.waitKey(0)


def main():
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            post_process(results, frame)

            annotated_frame = results[0].plot()

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
            # print("here!", box, cls, class_names[cls], i)
            # p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            # an_frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

            # x_min, y_min = max(0, p1[0]), max(
            #     0, p1[1]
            # )  # Ensure coordinates stay within frame bounds
            # x_max, y_max = min(frame.shape[1], p2[0]), min(frame.shape[0], p2[1])

            # # Crop the frame based on extracted coordinates
            # cropped_image = frame[y_min:y_max, x_min:x_max]

            # cv2.imshow("smth", cropped_image)
            # key = cv2.waitKey(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)
