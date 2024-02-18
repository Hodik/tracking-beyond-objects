def euclidean_distance(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    center_x1 = (x1 + x2) / 2
    center_y1 = (y1 + y2) / 2
    center_x2 = (x3 + x4) / 2
    center_y2 = (y3 + y4) / 2
    return ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    # Calculate area of intersection
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(
        0, min(y2, y4) - max(y1, y3)
    )
    # Calculate area of each box
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    # Calculate IoU
    return intersection_area / (box1_area + box2_area - intersection_area)


def distance_with_iou(box1, box2):
    if iou(box1, box2) > 0:
        return 0
    return euclidean_distance(box1, box2)
