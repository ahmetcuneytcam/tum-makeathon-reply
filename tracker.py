import collections
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import motmetrics as mm

mm.lap.default_solver = "lap"

from distance_metrics import compute_distance_matrix

def ltrb_to_ltwh(ltrb_boxes):
    ltwh_boxes = copy.deepcopy(ltrb_boxes)
    ltwh_boxes[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    ltwh_boxes[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]

    return ltwh_boxes


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, box, score, track_id, feature=None, inactive=0):
        self.id = track_id
        self.box = box
        self.score = score
        self.feature = collections.deque([feature])
        self.inactive = inactive
        self.max_features_num = 10

    def add_feature(self, feature):
        """Adds new appearance features to the object."""
        self.feature.append(feature)
        if len(self.feature) > self.max_features_num:
            self.feature.popleft()

    def get_feature(self):
        if len(self.feature) > 1:
            feature = torch.stack(list(self.feature), dim=0)
        else:
            feature = self.feature[0].unsqueeze(0)
        return feature.mean(0, keepdim=False)

class Tracker:
    """The baseclass for trackers"""

    def __init__(self, obj_detect):
        self.obj_detect = obj_detect

        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.mot_accum = None

    def reset(self, hard=True):
        self.tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(new_boxes[i], new_scores[i], self.track_num + i))
        self.track_num += num_new

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            device = torch.device("cuda:0" if next(self.obj_detect.parameters()).is_cuda else "cpu")
            box = torch.zeros(0).to(device)
        return box

    def data_association(self, boxes, scores):
        raise NotImplementedError

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        # boxes, scores = self.obj_detect.detect(frame)
        preds = self.obj_detect(frame)[0]
        preds = {k: v.cpu() for k, v in preds.items()}
        boxes = preds["boxes"].detach() # (l,t,r,b)
        # boxes = boxes[:, [1, 0, 3, 2]] 
        scores = preds["scores"].detach()
        # print(boxes)
        

        # data association
        self.data_association(boxes, scores)

        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.detach().cpu().numpy(), np.array([t.score.detach()])])

        self.im_index += 1

    def get_results(self):
        return self.results


# IOU BASED NOT WORKING

# def IoU(box1, box2):
#     """Computes the IoU between two boxes.
#     Args:
#         box1: The first box.
#         box2: The second box.
#     """
#     l = max(box1[0], box2[0])
#     t = max(box1[1], box2[1])
#     r = min(box1[2], box2[2])
#     b = min(box1[3], box2[3])

#     inter_area = max(0, r - l) * max(0, b - t)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou

# def iou_matrix(track_boxes, boxes):
#     """Computes the IoU matrix between two sets of boxes.
#     Args:
#         track_boxes: The first set of boxes.
#         boxes: The second set of boxes.
#     """
#     matrix = np.zeros((len(track_boxes), len(boxes)))
#     for i, track_box in enumerate(track_boxes):
#         for j, box in enumerate(boxes):
#             matrix[i, j] = IoU(track_box, box)
#     return matrix


# FALL BACK TO DISTANCE METRICS




# Old Tracker
class TrackerIoU(Tracker):

    def get_distance(self, boxes):

        # Euclidean 
        track_boxes = np.stack([t.box.detach().numpy() for t in self.tracks], axis=0)

        # iou_track_boxes = ltrb_to_ltwh(track_boxes)
        # iou_boxes = ltrb_to_ltwh(boxes)
        boxes = boxes.detach().cpu().numpy()
        track_centers = (track_boxes[:, 0:2] + track_boxes[:, 2:4]) / 2
        boxes_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2

        distance = np.zeros((len(track_boxes), len(boxes)))
        for i, track_box in enumerate(track_centers):
            for j, box in enumerate(boxes_centers):
                distance[i, j] = np.linalg.norm(track_box - box)

        # IoU
        # track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)

        # iou_track_boxes = ltrb_to_ltwh(track_boxes)
        # iou_boxes = ltrb_to_ltwh(boxes)
        # distance = mm.distances.iou_matrix(iou_track_boxes, iou_boxes.numpy(), max_iou=0.5)

        return distance


class TrackerIoUReID(TrackerIoU):
    def __init__(self, obj_detect, reid):
        super().__init__(obj_detect)
        self.reid = reid

    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them. Also store appearance features. """
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(new_boxes[i], new_scores[i], self.track_num + i, new_features[i]))
        self.track_num += num_new

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame)

        # reid feature extraction
        crops = self.get_crop_from_boxes(boxes, frame)
        pred_features = self.compute_reid_features(crops).cpu().clone()

        self.data_association(boxes, scores, pred_features)

        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1

    def get_crop_from_boxes(self, boxes, frame, height=256, width=128):
        """Crops all persons from a frame given the boxes.

        Args:
                boxes: The bounding boxes.
                frame: The current frame.
                height (int, optional): [description]. Defaults to 256.
                width (int, optional): [description]. Defaults to 128.
        """
        person_crops = []
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std
        for box in boxes:
            box = box.to(torch.int32)
            res = frame[:, :, box[1] : box[3], box[0] : box[2]]
            res = F.interpolate(res, (height, width), mode="bilinear")
            res = TF.normalize(res[0, ...], norm_mean, norm_std)
            person_crops.append(res.unsqueeze(0))

        return person_crops

    def compute_reid_features(self, crops):
        f_ = []
        self.reid.eval()
        device = torch.device("cuda:0" if next(self.reid.parameters()).is_cuda else "cpu")
        with torch.no_grad():
            for data in crops:
                img = data.to(device)
                features = self.reid(img)
                features = features.cpu().clone()
                f_.append(features)
            f_ = torch.cat(f_, 0)
            return f_

    def get_app_distance(self, pred_features, metric_fn):
        track_features = torch.stack([t.get_feature() for t in self.tracks], dim=0)
        appearance_distance = compute_distance_matrix(track_features, pred_features, metric_fn=metric_fn)
        appearance_distance = appearance_distance.numpy() * 0.5
        return appearance_distance
