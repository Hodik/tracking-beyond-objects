class ObjectTracker:
    def __init__(self, _id, initial_box, cls_id, people, objects):
        self._id = _id
        self.boxes = [initial_box]
        self.cls_id = cls_id
        self.related_objects: dict[int, int] = {}
        self.people = people
        self.objects = objects

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

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, __value: object) -> bool:
        return self._id == __value._id

    def __str__(self) -> str:
        return f"ObjectTracker {self._id} class <{self.cls_id}>, related: {self.related_objects}"


class PersonTracker(ObjectTracker): ...
