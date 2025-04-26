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

            # print(track_ids, len(track_ids))
            # print(distance, distance.shape)
            # print(row_idx, len(row_idx))
            # print(col_idx, len(col_idx), "\n")
            
            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.

            ########################################################################
            # TODO:                                                                #
            # Update existing tracks and remove unmatched tracks.                  #
            # Reminder: If the costs are equal to _UNMATCHED_COST, it's NOT a      #
            # match. Be careful with overriding self.tracks, as past tracks will   #
            # be gone.                                                             #
            #                                                                      #
            # NOTE 1: self.tracks = ... <-- needs to be filled.                    #
            #                                                                      #
            # NOTE 2: # 1. costs == _UNMATCHED_COST -> remove.                     #
            # Optional: 2. tracks that have no match -> remove.                    #
            #                                                                      #
            # NOTE 3: Add new tracks. See TrackerIoU.                              #
            # new_boxes = []  # <-- needs to be filled.                            #
            # new_scores = []  # <-- needs to be filled.                           #
            ########################################################################

            # unmatched_track_ids = []
            # for t, dist in zip(self.tracks, distance):
            #     if np.all(dist == _UNMATCHED_COST):
            #         unmatched_track_ids.append(t.id)                                        # add unmatched tracks
            #     elif t.id in row_idx:
            #         match_id = col_idx[row_idx == t.id].item()                              # get the match id according to "linear_assignment"
            #         t.box = boxes[match_id]                                                 # update the box of the track
            # self.tracks = [t for t in self.tracks if t.id not in unmatched_track_ids]       # remove unmatched tracks

            # # add new tracks
            # new_boxes = []
            # new_scores = []
            # for i, dist in enumerate(np.transpose(distance)):
            #     if np.all(dist == _UNMATCHED_COST):                                         # check if some detections are unmatched
            #         new_boxes.append(boxes[i])                                              # add unmatched boxes 
            #         new_scores.append(scores[i])                                            # add unmatched scores

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

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################

            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)


# class Hungarian_TrackerIoUReID(TrackerIoUReID):
#     name = "Hungarian_TrackerIoUReID"

#     def compute_distance_matrix(self, distance_app, distance_iou, alpha=0.0):
#         UNMATCHED_COST = 255.0
#         # Build cost matrix.
#         assert np.alltrue(distance_app >= -0.1)
#         assert np.alltrue(distance_app <= 1.1)

#         combined_costs = alpha * distance_iou + (1 - alpha) * distance_app

#         # Set all unmatched costs to _UNMATCHED_COST.
#         distance = np.where(np.isnan(distance_iou), UNMATCHED_COST, combined_costs)
#         return distance

#     def data_association(self, boxes, scores, pred_features):

#         if self.tracks:
#             track_ids = [t.id for t in self.tracks]
        
#             distance_iou = self.get_iou_distance(boxes)
#             distance_app = self.get_app_distance(pred_features, metric_fn=cosine_distance) # This will use your similarity measure. Please use cosine_distance!
#             distance = self.compute_distance_matrix(
#                 distance_app, distance_iou,
#             )

#             # Perform Hungarian matching.
#             row_idx, col_idx = linear_assignment(distance)

#             # row_idx and col_idx are indices into track_boxes and boxes.
#             # row_idx[i] and col_idx[i] define a match.
#             # distance[row_idx[i], col_idx[i]] define the cost for that matching.

#             ########################################################################
#             # TODO:                                                                #
#             # Update existing tracks and remove unmatched tracks.                  #
#             # Reminder: If the costs are equal to _UNMATCHED_COST, it's NOT a      #
#             # match. Be careful with overriding self.tracks, as past tracks will   #
#             # be gone.                                                             #
#             #                                                                      #
#             # NOTE: Please update the feature of a track by using add_feature:     #
#             # self.tracks[my_track_id].add_feature(pred_features[my_feat_index])   #
#             # Reason: We use the mean feature from the last 10 frames for ReID.    #
#             #                                                                      #
#             # NOTE 1: self.tracks = ... <-- needs to be filled.                    #
#             #                                                                      #
#             # NOTE 2: # 1. costs == _UNMATCHED_COST -> remove.                     #
#             # Optional: 2. tracks that have no match -> remove.                    #
#             #                                                                      #
#             # NOTE 3: Add new tracks. See TrackerIoU.                              #
#             # new_boxes = []  # <-- needs to be filled.                            #
#             # new_scores = []  # <-- needs to be filled.                           #
#             ########################################################################

#             # # Update existing tracks and identify unmatched (without good enough match) tracks.
#             # unmatched_track_ids = []
#             # for t_id, dist_row in zip(track_ids, distance):
#             #     if np.all(dist_row == _UNMATCHED_COST):
#             #         unmatched_track_ids.append(t_id)
#             #     elif len(col_idx[row_idx == t_id])!=0:
#             #         match_id = col_idx[row_idx == t_id]
#             #         assert len(match_id) <= 1, "Multiple matches for one track"
#             #         assert len(match_id) > 0, "No match for track"
#             #         self.tracks[t_id].box = boxes[match_id[0]]
#             #         self.tracks[t_id].add_feature(pred_features[match_id[0]])
            
#             # # Identify lost (undetected) tracks, that is when num tracks > num detections
#             # # unmatched_track_ids.extend([t_id for t_id in track_ids if t_id not in row_idx]) # add tracks that have no match according to "linear_assignment"

#             # # remove unmatched tracks
#             # self.tracks = [t for t in self.tracks if t.id not in unmatched_track_ids]

#             # # Add new tracks
#             # new_boxes = []
#             # new_scores = []
#             # new_features = []
#             # for i, dist_row in enumerate(distance.T):
#             #     if np.all(dist_row == _UNMATCHED_COST):
#             #         new_boxes.append(boxes[i])
#             #         new_scores.append(scores[i])
#             #         new_features.append(pred_features[i])
            
#             # Identify matches and unmatched entities
#             matched_tracks = set()
#             unmatched_tracks = set(track_ids)
#             unmatched_detections = set(range(len(boxes)))

#             for r, c in zip(row_idx, col_idx):
#                 if distance[r, c] < _UNMATCHED_COST:  # Valid match
#                     self.tracks[r].box = boxes[c]
#                     self.tracks[r].add_feature(pred_features[c])
#                     matched_tracks.add(self.tracks[r].id)
#                     unmatched_detections.discard(c)

#             unmatched_tracks.difference_update(matched_tracks)

#             # Remove unmatched tracks
#             self.tracks = [t for t in self.tracks if t.id not in unmatched_tracks]

#             # Add new tracks for unmatched detections
#             new_boxes = [boxes[i] for i in unmatched_detections]
#             new_scores = [scores[i] for i in unmatched_detections]
#             new_features = [pred_features[i] for i in unmatched_detections]

#             pass

#             ########################################################################
#             #                           END OF YOUR CODE                           #
#             ########################################################################

#             self.add(new_boxes, new_scores, new_features)
#         else:
#             # No tracks exist.
#             self.add(boxes, scores, pred_features)
