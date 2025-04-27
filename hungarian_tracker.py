import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from distance_metrics import cosine_distance
from tracker import TrackerIoU, TrackerIoUReID

_UNMATCHED_COST = 255.0

# # Old Tracker
# class Min_TrackerIoU(TrackerIoU):
#     name = "Min_TrackerIoU"

#     def data_association(self, boxes, scores):
#         if self.tracks:
#             distance = self.get_iou_distance(boxes)

#             # update existing tracks
#             remove_track_ids = []
#             for t, dist in zip(self.tracks, distance):
#                 if np.isnan(dist).all():
#                     remove_track_ids.append(t.id)
#                 else:
#                     match_id = np.nanargmin(dist)
#                     t.box = boxes[match_id]
#             self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

#             # add new tracks
#             new_boxes = []
#             new_scores = []
#             for i, dist in enumerate(np.transpose(distance)):
#                 if np.isnan(dist).all():
#                     new_boxes.append(boxes[i])
#                     new_scores.append(scores[i])
#             self.add(new_boxes, new_scores)

#         else:
#             self.add(boxes, scores)

class Hungarian_TrackerIoU(TrackerIoU):
    name = "Hungarian_TrackerIoU"

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            # print("boxes",boxes)

            distance = self.get_distance(boxes)
            # print("distance",distance)
            # print(distance, distance.shape)
            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), _UNMATCHED_COST, distance)
            # if not np.all(distance == 0):
            #     if not np.all(distance == _UNMATCHED_COST):
            #         print(distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)

            # Identify matches and unmatched tracks
            matched_tracks = set()
            unmatched_tracks = set(track_ids)
            unmatched_detections = set(range(len(boxes)))

            for r, c in zip(row_idx, col_idx):
                if distance[r, c] < _UNMATCHED_COST:  # Valid match
                    self.tracks[r].box = boxes[c]
                    matched_tracks.add(self.tracks[r].id)
                    unmatched_detections.discard(c)

            unmatched_tracks.difference_update(matched_tracks)

            # Remove unmatched tracks
            self.tracks = [t for t in self.tracks if t.id not in unmatched_tracks]

            # Add new tracks for unmatched detections
            new_boxes = [boxes[i] for i in unmatched_detections]
            # print("new",new_boxes)
            new_scores = [scores[i] for i in unmatched_detections]

            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)
