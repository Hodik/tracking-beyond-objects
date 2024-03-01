from copy import deepcopy

import torch

from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Results


def plot(
    results: Results,
    actions: dict = {},
    conf=True,
    line_width=None,
    font_size=None,
    font="Arial.ttf",
    pil=False,
    labels=True,
    boxes=True,
    probs=True,
):
    names = results.names
    is_obb = results.obb is not None
    pred_boxes, show_boxes = results.obb if is_obb else results.boxes, boxes
    pred_probs, show_probs = results.probs, probs
    annotator = Annotator(
        deepcopy(results.orig_img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = "" if id is None else f"id:{id} "
            name += ",".join(a[0] for a in actions.get(id)) if actions.get(id, None) else names[c]
            label = (f"{name} {conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

    return annotator.result()