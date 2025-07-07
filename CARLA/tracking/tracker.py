"""
tracker.py
----------
LiDAR-based 3D multi-object tracker using Kalman filters and DBSCAN.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import uuid

class Track3D:
    def __init__(self, initial_position, dt=0.1):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.eye(3, 6)
        self.kf.R *= 0.1
        self.kf.P *= 10
        self.kf.Q *= 0.01
        self.kf.x[:3] = np.array(initial_position).reshape(3, 1)

        self.id = str(uuid.uuid4())[:8]
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        self.kf.update(np.array(detection).reshape(3, 1))
        self.time_since_update = 0

    def get_position(self):
        return self.kf.x[:3].flatten()

    def get_velocity(self):
        return self.kf.x[3:].flatten()


class Tracker3D:
    def __init__(self, dt=0.1, eps=1.2, min_samples=3, max_age=5, dist_threshold=2.5):
        self.dt = dt
        self.eps = eps
        self.min_samples = min_samples
        self.max_age = max_age
        self.dist_threshold = dist_threshold
        self.tracks = []

    def update(self, points_3d):
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_3d)
        labels = clustering.labels_
        centers = [points_3d[labels == label].mean(axis=0) for label in set(labels) if label != -1]

        for track in self.tracks:
            track.predict()

        matched, unmatched_dets = [], list(range(len(centers)))

        if self.tracks:
            cost = np.zeros((len(self.tracks), len(centers)))
            for i, track in enumerate(self.tracks):
                for j, center in enumerate(centers):
                    cost[i, j] = np.linalg.norm(track.get_position() - center)

            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < self.dist_threshold:
                    matched.append((r, c))
                    unmatched_dets.remove(c)

        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(centers[det_idx])

        for det_idx in unmatched_dets:
            self.tracks.append(Track3D(centers[det_idx], dt=self.dt))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [(t.id, t.get_position(), t.get_velocity()) for t in self.tracks]
