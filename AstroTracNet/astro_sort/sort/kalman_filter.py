# vim: expandtab:ts=4:sw=4
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


class SimpleStableKalmanFilter(KalmanFilter):
    """
    Enhanced Kalman filter that improves stability and reduces ID switches
    through velocity consistency and appearance feature matching.
    """
    
    def __init__(self, max_history_length: int = 60, velocity_weight: float = 0.5, 
                appearance_weight: float = 0.7, time_since_update_threshold: int = 15):
        """
        Initialize the enhanced Kalman filter with stability parameters.
        
        Parameters
        ----------
        max_history_length : int
            Maximum length of track history to maintain
        velocity_weight : float
            Weight for velocity consistency (0-1)
        appearance_weight : float
            Weight for appearance features in matching (0-1)
        time_since_update_threshold : int
            Base threshold for how many frames a track can survive without updates
        """
        super(SimpleStableKalmanFilter, self).__init__()
        
        # Core parameters
        self.max_history_length = max_history_length
        self.velocity_weight = velocity_weight
        self.appearance_weight = appearance_weight
        self.time_since_update_threshold = time_since_update_threshold
        
        # State records
        self.history: Dict[int, List[np.ndarray]] = {}  # Track history
        self.velocities: Dict[int, List[np.ndarray]] = {}  # Velocity history
        self.appearances: Dict[int, List[np.ndarray]] = {}  # Appearance features
        self.last_update_time: Dict[int, int] = {}  # Last update frame
        self.current_frame = 0  # Current frame counter
        
        # Track parameters
        self.track_quality: Dict[int, float] = {}  # Track quality score (0-1)
        self.occlusion_counts: Dict[int, int] = {}  # Occlusion counter
        self.inactive_counts: Dict[int, int] = {}  # Inactivity counter
        
        # Improved process noise parameters - reduced randomness
        self._std_weight_position = 1. / 25
        self._std_weight_velocity = 1. / 180
        
        # Thresholds
        self.SIMILARITY_THRESHOLD = 0.7
        self.APPEARANCE_UPDATE_THRESHOLD = 0.7
        self.OCCLUSION_FRAME_GAP = 1
        self.QUALITY_BONUS_FACTOR = 10
        self.HIGH_QUALITY_THRESHOLD = 0.7
        self.LONG_TRACK_THRESHOLD = 30
        self.INACTIVE_QUALITY_DECAY = 0.95
        self.QUALITY_INCREASE_STEP = 0.05
        self.QUALITY_DECREASE_STEP = 0.9
        self.SIMILARITY_BONUS_FACTOR = 10
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray, 
                track_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced prediction step with velocity smoothing.
        
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state
        track_id : Optional[int]
            ID of the track for accessing historical data
            
        Returns
        -------
        (ndarray, ndarray)
            Returns the predicted state's mean vector and covariance matrix
        """
        # Standard Kalman prediction
        mean_pred, covariance_pred = super().predict(mean, covariance)
        
        # Apply velocity smoothing from history if available
        if track_id is not None and track_id in self.velocities and len(self.velocities[track_id]) > 2:
            # Calculate weighted average velocity, favoring recent velocities
            weights = np.linspace(0.5, 1.0, min(5, len(self.velocities[track_id])))
            weights /= weights.sum()
            recent_velocities = self.velocities[track_id][-min(5, len(self.velocities[track_id])):]
            
            # Calculate weighted average velocity
            avg_velocity = np.zeros(4)
            for i, v in enumerate(recent_velocities[-len(weights):]):
                avg_velocity += weights[i] * v
            
            # Smooth fusion of current prediction and historical velocity
            current_velocity = mean_pred[4:8]
            smoothed_velocity = self.velocity_weight * current_velocity + (1 - self.velocity_weight) * avg_velocity
            
            # Apply smoothed velocity
            mean_pred[4:8] = smoothed_velocity
        
        return mean_pred, covariance_pred
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, 
              track_id: Optional[int] = None, appearance_feature: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step with track history and appearance feature management.
        
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional)
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional)
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h)
        track_id : Optional[int]
            ID of the track for updating history
        appearance_feature : Optional[ndarray]
            Feature vector representing the object's appearance
            
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution
        """
        # Execute standard Kalman update
        new_mean, new_covariance = super().update(mean, covariance, measurement)
        
        # Update track history
        if track_id is not None:
            # First appearance
            if track_id not in self.history:
                self.history[track_id] = []
                self.velocities[track_id] = []
                self.appearances[track_id] = []
                self.track_quality[track_id] = 0.5  # Initial quality is medium
                self.occlusion_counts[track_id] = 0
                self.inactive_counts[track_id] = 0
            
            # Detect potential occlusions
            if track_id in self.last_update_time:
                gap = self.current_frame - self.last_update_time[track_id]
                if gap > self.OCCLUSION_FRAME_GAP:  # Frame gap detected, potential occlusion
                    self.occlusion_counts[track_id] += 1
                    # Reduce track quality
                    self.track_quality[track_id] *= self.QUALITY_DECREASE_STEP
                else:
                    # Increase quality with continuous tracking
                    self.track_quality[track_id] = min(1.0, self.track_quality[track_id] + self.QUALITY_INCREASE_STEP)
                
                # Reset inactivity counter
                self.inactive_counts[track_id] = 0
            
            # Update history state and velocity
            self.history[track_id].append(new_mean)
            
            # Limit history length to prevent memory leaks
            if len(self.history[track_id]) > self.max_history_length:
                self.history[track_id].pop(0)
            
            # Update velocity history
            if len(self.history[track_id]) >= 2:
                new_velocity = new_mean[4:8]
                self.velocities[track_id].append(new_velocity)
                if len(self.velocities[track_id]) > self.max_history_length:
                    self.velocities[track_id].pop(0)
            
            # Update appearance features
            if appearance_feature is not None:
                # Directly add if not enough appearance history
                if len(self.appearances[track_id]) < 5:
                    self.appearances[track_id].append(appearance_feature)
                else:
                    # Calculate similarity with existing features
                    similarities = []
                    for feat in self.appearances[track_id]:
                        norm_product = np.linalg.norm(feat) * np.linalg.norm(appearance_feature) + 1e-6
                        sim = np.dot(feat, appearance_feature) / norm_product
                        similarities.append(sim)
                    
                    avg_sim = np.mean(similarities)
                    
                    # Only update history when new feature is sufficiently similar
                    if avg_sim > self.APPEARANCE_UPDATE_THRESHOLD:
                        # Update appearance features using sliding average
                        self.appearances[track_id].append(appearance_feature)
                        if len(self.appearances[track_id]) > 10:  # Limit appearance history
                            self.appearances[track_id].pop(0)
            
            # Update last update time
            self.last_update_time[track_id] = self.current_frame
        
        return new_mean, new_covariance
    
    def compute_similarity_score(self, track_id: int, appearance_feature: np.ndarray) -> float:
        """
        Compute similarity score with track's historical appearances.
        
        Parameters
        ----------
        track_id : int
            ID of the track
        appearance_feature : ndarray
            Feature vector to compare against track history
            
        Returns
        -------
        float
            Similarity score (0-1) with higher values meaning more similar
        """
        if track_id not in self.appearances or not self.appearances[track_id] or appearance_feature is None:
            return 0.5  # Default medium similarity
        
        # Calculate cosine similarity with all historical features
        similarities = []
        for feat in self.appearances[track_id]:
            norm_product = np.linalg.norm(feat) * np.linalg.norm(appearance_feature) + 1e-6
            sim = np.dot(feat, appearance_feature) / norm_product
            similarities.append(sim)
        
        # Return weighted average similarity, giving higher weight to recent features
        weights = np.linspace(0.5, 1.0, len(similarities))
        weights /= weights.sum()
        weighted_sim = np.sum(weights * np.array(similarities))
        
        return weighted_sim
    
    def get_combined_distance(self, mahalanobis_distance: float, appearance_distance: float, 
                              track_id: int) -> float:
        """
        Weighted fusion of Mahalanobis distance and appearance distance.
        
        Parameters
        ----------
        mahalanobis_distance : float
            Mahalanobis distance from Kalman filter
        appearance_distance : float
            Appearance distance (1-similarity)
        track_id : int
            ID of the track for quality and history factors
            
        Returns
        -------
        float
            Combined distance metric for assignment
        """
        # Adjust weights based on track quality and history
        quality_factor = self.track_quality.get(track_id, 0.5)
        time_since_update = self.current_frame - self.last_update_time.get(track_id, 0)
        
        # Tracks with long disappearance should rely more on appearance
        if time_since_update > 5:
            appearance_factor = min(0.9, self.appearance_weight + 0.1 * time_since_update / 10.0)
        else:
            appearance_factor = self.appearance_weight
        
        # High quality tracks rely more on Mahalanobis distance
        appearance_factor = appearance_factor * quality_factor
        mahalanobis_factor = 1.0 - appearance_factor
        
        # Weighted fusion
        combined_distance = mahalanobis_factor * mahalanobis_distance + appearance_factor * appearance_distance
        return combined_distance
    
    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, 
                        measurements: np.ndarray, only_position: bool = False, 
                        track_id: Optional[int] = None, 
                        appearance_features: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Enhanced gating distance calculation, fusing Mahalanobis distance and appearance distance.
        
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional)
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional)
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements
        only_position : bool
            If True, use only position for distance computation
        track_id : Optional[int]
            ID of the track for accessing historical data
        appearance_features : Optional[List[ndarray]]
            List of appearance features for each measurement
            
        Returns
        -------
        ndarray
            Array of distances for each measurement
        """
        # Standard Mahalanobis distance
        maha_distances = super().gating_distance(mean, covariance, measurements, only_position)
        
        # Return only Mahalanobis distance if no appearance data
        if appearance_features is None or track_id is None or track_id not in self.appearances:
            return maha_distances
        
        # Create appearance distance matrix
        appearance_distances = np.zeros_like(maha_distances)
        
        # Calculate appearance distance (1-similarity)
        for i, feature in enumerate(appearance_features):
            similarity = self.compute_similarity_score(track_id, feature)
            appearance_distances[i] = 1.0 - similarity
        
        # Combine Mahalanobis distance and appearance distance
        combined_distances = np.zeros_like(maha_distances)
        for i in range(len(measurements)):
            combined_distances[i] = self.get_combined_distance(
                maha_distances[i], appearance_distances[i], track_id)
        
        return combined_distances
    
    def increment_frame(self) -> None:
        """
        Increment frame counter and update all track states.
        Updates track quality scores and marks inactive tracks for cleanup.
        """
        self.current_frame += 1
        
        # Update all inactive tracks
        inactive_tracks = []
        for track_id in self.history:
            if track_id not in self.last_update_time or self.current_frame - self.last_update_time[track_id] > 0:
                if track_id not in self.inactive_counts:
                    self.inactive_counts[track_id] = 0
                self.inactive_counts[track_id] += 1
                
                # Decrease quality for long inactive tracks
                if self.inactive_counts[track_id] > 5:
                    self.track_quality[track_id] *= self.INACTIVE_QUALITY_DECAY
                
                # Mark for cleanup tracks inactive for too long
                if self.inactive_counts[track_id] > self.max_history_length:
                    inactive_tracks.append(track_id)
        
        # Clean up old track data
        for track_id in inactive_tracks:
            self._clean_track_data(track_id)
    
    def _clean_track_data(self, track_id: int) -> None:
        """
        Clean up track data to prevent memory leaks.
        
        Parameters
        ----------
        track_id : int
            ID of the track to clean up
        """
        for storage in [self.history, self.velocities, self.appearances, 
                        self.track_quality, self.occlusion_counts, 
                        self.inactive_counts, self.last_update_time]:
            if track_id in storage:
                del storage[track_id]
    
    def should_keep_track(self, track_id: int, 
                          new_detection_features: Optional[np.ndarray] = None) -> bool:
        """
        Determine if a track should be kept (handling short occlusions).
        
        Parameters
        ----------
        track_id : int
            ID of the track
        new_detection_features : Optional[ndarray]
            Features from new detections to compare with track history
            
        Returns
        -------
        bool
            True if the track should be maintained, False if it should be deleted
        """
        if track_id not in self.last_update_time:
            return False
        
        time_since_update = self.current_frame - self.last_update_time[track_id]
        
        # Basic condition: keep tracks within time threshold
        keep_based_on_time = time_since_update <= self.time_since_update_threshold
        
        # Additional condition: high quality tracks get longer retention
        quality_bonus = int(self.QUALITY_BONUS_FACTOR * self.track_quality.get(track_id, 0))
        extended_threshold = self.time_since_update_threshold + quality_bonus
        
        # Use extended threshold for long-lived high-quality tracks
        track_age = len(self.history.get(track_id, []))
        if track_age > self.LONG_TRACK_THRESHOLD and self.track_quality.get(track_id, 0) > self.HIGH_QUALITY_THRESHOLD:
            keep_based_on_extended_time = time_since_update <= extended_threshold
        else:
            keep_based_on_extended_time = False
        
        # Appearance matching condition
        keep_based_on_similarity = False
        if new_detection_features is not None and track_id in self.appearances:
            best_similarity = 0
            for feat in self.appearances[track_id]:
                norm_product = np.linalg.norm(feat) * np.linalg.norm(new_detection_features) + 1e-6
                sim = np.dot(feat, new_detection_features) / norm_product
                best_similarity = max(best_similarity, sim)
            
            # High similarity can tolerate longer disappearance
            if best_similarity > self.SIMILARITY_THRESHOLD:
                # Additional grace period based on similarity
                sim_factor = (best_similarity - self.SIMILARITY_THRESHOLD) / (1 - self.SIMILARITY_THRESHOLD)
                similarity_bonus = int(self.SIMILARITY_BONUS_FACTOR * sim_factor)
                keep_based_on_similarity = time_since_update <= (extended_threshold + similarity_bonus)
        
        # Combined conditions
        return keep_based_on_time or keep_based_on_extended_time or keep_based_on_similarity
