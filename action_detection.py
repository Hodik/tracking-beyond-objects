import numpy as np
import torch
from mmengine.structures import InstanceData

from mmaction2.mmaction.structures import ActionDataSample

from mmdet.apis import init_detector


class TaskInfo:
    """Wapper for a clip.

    Transmit data around three threads.

    1) Read Thread: Create task and put task into read queue. Init `frames`,
        `processed_frames`, `img_shape`, `ratio`, `clip_vis_length`.
    2) Main Thread: Get data from read queue, predict human bboxes and stdet
        action labels, draw predictions and put task into display queue. Init
        `display_bboxes`, `stdet_bboxes` and `action_preds`, update `frames`.
    3) Display Thread: Get data from display queue, show/write frames and
        delete task.
    """

    def __init__(self, clip_vis_length = -1, frames_inds = None, ratio = None):
        self.id = -1

        # raw frames, used as human detector input, draw predictions input
        # and output, display input
        self.frames = None

        # stdet params
        self.processed_frames = None  # model inputs
        self.frames_inds = frames_inds  # select frames from processed frames
        self.img_shape = None  # model inputs, processed frame shape
        # `action_preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        self.action_preds = None  # stdet results

        # human bboxes with the format (xmin, ymin, xmax, ymax)
        self.display_bboxes = None  # bboxes coords for self.frames
        self.stdet_bboxes = None  # bboxes coords for self.processed_frames
        self.ratio = ratio  # processed_frames.shape[1::-1]/frames.shape[1::-1]

        # for each clip, draw predictions on clip_vis_length frames
        self.clip_vis_length = clip_vis_length
        self.keyframe = None

    def add_frames(self, idx, frames, processed_frames):
        """Add the clip and corresponding id.

        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
            processed_frames (list[ndarray]): list of resize and normed images
                in "BGR" format.
        """
        self.frames = frames
        self.processed_frames = processed_frames
        self.id = idx
        self.img_shape = processed_frames[0].shape[:2]

    def add_bboxes(self, display_bboxes):
        """Add correspondding bounding boxes."""
        self.display_bboxes = display_bboxes
        self.stdet_bboxes = display_bboxes.clone()
        self.stdet_bboxes[:, ::2] = self.stdet_bboxes[:, ::2] * self.ratio[0]
        self.stdet_bboxes[:, 1::2] = self.stdet_bboxes[:, 1::2] * self.ratio[1]

    def add_action_preds(self, preds):
        """Add the corresponding action predictions."""
        self.action_preds = preds

    def get_model_inputs(self, device):
        """Convert preprocessed images to MMAction2 STDet model inputs."""
        cur_frames = [self.processed_frames[idx] for idx in self.frames_inds]
        input_array = np.stack(cur_frames).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(device)
        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=self.stdet_bboxes)
        datasample.set_metainfo(dict(img_shape=self.img_shape))

        return dict(inputs=input_tensor, data_samples=[datasample], mode="predict")


class StdetPredictor:
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (str): Path to stdet config.
        ckpt (str): Path to stdet checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self, config, checkpoint, device, score_thr, label_map_path):
        self.score_thr = score_thr

        # load model
        config.model.backbone.pretrained = None
        # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        # load_checkpoint(model, checkpoint, map_location='cpu')
        # model.to(device)
        # model.eval()
        model = init_detector(config, checkpoint, device=device)
        self.model = model
        self.device = device

        # init label map, aka class_id to class_name dict
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(": ") for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}
        try:
            if config["data"]["train"]["custom_classes"] is not None:
                self.label_map = {
                    id + 1: self.label_map[cls]
                    for id, cls in enumerate(config["data"]["train"]["custom_classes"])
                }
        except KeyError:
            pass

    def predict(self, task):
        """Spatio-temporval Action Detection model inference."""
        # No need to do inference if no one in keyframe
        if len(task.stdet_bboxes) == 0:
            return task

        with torch.no_grad():
            result = self.model(**task.get_model_inputs(self.device))
        scores = result[0].pred_instances.scores
        # pack results of human detector and stdet
        preds = []
        for _ in range(task.stdet_bboxes.shape[0]):
            preds.append([])
        for class_id in range(scores.shape[1]):
            if class_id not in self.label_map:
                continue
            for bbox_id in range(task.stdet_bboxes.shape[0]):
                if scores[bbox_id][class_id] > self.score_thr:
                    preds[bbox_id].append(
                        (self.label_map[class_id], scores[bbox_id][class_id].item())
                    )

        # update task
        # `preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        task.add_action_preds(preds)

        return task
