# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade with hierarchical matching strategy.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    
    # 根据轨迹质量进行分类
    stable_tracks = []
    normal_tracks = []
    unstable_tracks = []
    
    for idx in track_indices:
        track = tracks[idx]
        # 高质量轨迹: 长时间被确认且没有太多丢失
        if track.is_confirmed() and track.hits > 10 and track.time_since_update <= 2:
            stable_tracks.append(idx)
        # 低质量轨迹: 确认时间短或最近丢失较多
        elif track.is_confirmed() and (track.hits < 5 or track.time_since_update > 1):
            unstable_tracks.append(idx)
        # 其他正常轨迹
        else:
            normal_tracks.append(idx)
    
    # 初始化匹配和未匹配结果
    matches = []
    unmatched_detections = detection_indices
    matched_track_indices = set()
    
    # 1. 首先匹配高质量轨迹 (使用较高阈值)
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
            
        # 当前级别的高质量轨迹
        stable_tracks_l = [
            k for k in stable_tracks
            if tracks[k].time_since_update == 1 + level
        ]
        
        if not stable_tracks_l:
            continue
            
        # 高质量轨迹可以使用更高的匹配阈值
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance * 1.1, tracks, detections,
            stable_tracks_l, unmatched_detections)
        
        matches += matches_l
        matched_track_indices.update(k for k, _ in matches_l)
    
    # 2. 然后匹配普通轨迹 (使用标准阈值)
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
            
        # 排除已匹配的轨迹
        available_normal_tracks = [k for k in normal_tracks if k not in matched_track_indices]
        
        # 当前级别的普通轨迹
        normal_tracks_l = [
            k for k in available_normal_tracks
            if tracks[k].time_since_update == 1 + level
        ]
        
        if not normal_tracks_l:
            continue
            
        # 使用标准阈值
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            normal_tracks_l, unmatched_detections)
        
        matches += matches_l
        matched_track_indices.update(k for k, _ in matches_l)
    
    # 3. 最后匹配低质量轨迹 (使用较低阈值)
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
            
        # 排除已匹配的轨迹
        available_unstable_tracks = [k for k in unstable_tracks if k not in matched_track_indices]
        
        # 当前级别的低质量轨迹
        unstable_tracks_l = [
            k for k in available_unstable_tracks
            if tracks[k].time_since_update == 1 + level
        ]
        
        if not unstable_tracks_l:
            continue
            
        # 低质量轨迹使用更严格的阈值，防止错误匹配
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance * 0.8, tracks, detections,
            unstable_tracks_l, unmatched_detections)
        
        matches += matches_l
        matched_track_indices.update(k for k, _ in matches_l)
    
    # 计算未匹配的轨迹
    unmatched_tracks = [k for k in track_indices if k not in matched_track_indices]
    
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    # 检查是否是SimpleStableKalmanFilter
    if hasattr(kf, 'compute_similarity_score'):  # 检测SimpleStableKalmanFilter的特有方法
        # 使用增强版gating_distance
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            track_id = track.track_id
            
            # 准备测量和外观特征
            measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
            appearance_features = [detections[i].feature for i in detection_indices]
            
            # 调用增强版gating_distance
            gating_distances = kf.gating_distance(
                track.mean, track.covariance, measurements, 
                track_id=track_id, 
                appearance_features=appearance_features)
            
            # 应用门控
            cost_matrix[row, gating_distances > kalman_filter.chi2inv95[4]] = gated_cost
            
            # 调整匹配成本
            valid_indices = gating_distances <= kalman_filter.chi2inv95[4]
            for col_idx, valid in enumerate(valid_indices):
                if valid:
                    # 获取测量索引
                    detection_idx = detection_indices[col_idx]
                    # 调整成本
                    if cost_matrix[row, col_idx] < 0.7:  # 只调整合理的匹配
                        cost_matrix[row, col_idx] *= max(0.1, min(1.0, gating_distances[col_idx] / 10.0))
                        
        return cost_matrix
    
    # 原始的gate_cost_matrix逻辑
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
