# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    use_simple_stable : bool
        Whether to use SimpleStableKalmanFilter instead of the standard KalmanFilter.
    max_history_length : int
        Maximum number of states to keep in history (only used if use_simple_stable=True).
    velocity_weight : float
        Weight for blending current velocity with historical average (only used if use_simple_stable=True).
    appearance_weight : float
        Weight for appearance features in matching (only used if use_simple_stable=True).
    time_since_update_threshold : int
        Number of frames to maintain ID during occlusion (only used if use_simple_stable=True).

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter or kalman_filter.SimpleStableKalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, 
                 use_simple_stable=False, max_history_length=30, 
                 velocity_weight=0.6, appearance_weight=0.7, 
                 time_since_update_threshold=5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_simple_stable = use_simple_stable

        # Choose Kalman filter type based on configuration
        if use_simple_stable:
            self.kf = kalman_filter.SimpleStableKalmanFilter(
                max_history_length=max_history_length,
                velocity_weight=velocity_weight,
                appearance_weight=appearance_weight,
                time_since_update_threshold=time_since_update_threshold
            )
        else:
            self.kf = kalman_filter.KalmanFilter()
            
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            
            # If using SimpleStable, update track with appearance feature
            if self.use_simple_stable and hasattr(detections[detection_idx], 'feature'):
                track_id = self.tracks[track_idx].track_id
                appearance_feature = detections[detection_idx].feature
                # The update method now handles history updates internally
                # No need for separate update_history call
                
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # Keep only valid tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # If using SimpleStable, check if we should keep some tracks
        if self.use_simple_stable:
            for track in self.tracks:
                if not track.is_confirmed() and self.kf.should_keep_track(track.track_id):
                    track._n_init = max(1, track._n_init - 1)  # Reduce initialization frames for faster recovery

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            
            # 标准gating
            if not self.use_simple_stable:
                cost_matrix = linear_assignment.gate_cost_matrix(
                    self.kf, cost_matrix, tracks, dets, track_indices,
                    detection_indices)
            else:
                # 使用增强版gating_distance直接计算融合距离
                for i, track_idx in enumerate(track_indices):
                    track_id = tracks[track_idx].track_id
                    mean = tracks[track_idx].mean
                    covariance = tracks[track_idx].covariance
                    
                    # 准备测量和外观特征
                    measurements = np.array([dets[j].to_xyah() for j in detection_indices])
                    appearance_features = [dets[j].feature for j in detection_indices]
                    
                    if len(measurements) > 0:
                        # 直接调用增强版gating_distance，并传递正确的参数
                        gating_distances = self.kf.gating_distance(
                            mean, covariance, measurements, 
                            track_id=track_id, 
                            appearance_features=appearance_features)
                        
                        # 将gating距离融合到成本矩阵中
                        for j in range(len(detection_indices)):
                            # 仅对合理的匹配调整成本
                            if cost_matrix[i, j] < 0.7:
                                cost_matrix[i, j] *= max(0.1, min(1.0, gating_distances[j] / 10.0))
            
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track_id = self._next_id
        self.tracks.append(Track(
            mean, covariance, track_id, self.n_init, self.max_age,
            detection.feature))
            
        # If using SimpleStable, initialize track appearance feature
        if self.use_simple_stable and hasattr(detection, 'feature'):
            # The appearance features will be handled in the filter's update method
            # when called in Track.update() next time
            pass
            
        self._next_id += 1
