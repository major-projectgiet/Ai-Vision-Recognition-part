import math


class FaceTracker:
    def __init__(self, max_distance=80, max_missed_frames=10):
        self.next_track_id = 1
        self.tracks = {}

        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

    def _get_center(self, bbox):
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)

    def _calculate_distance(self, point1, point2):
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )

    def update(self, detections):
        """
        Update tracker using the faces detected in the current frame.

        detections:
            List of bounding boxes.
            Each bounding box is:
            (x, y, width, height)

        Returns:
            List of dictionaries containing:
            {
                "track_id": int,
                "bbox": tuple,
                "center": tuple
            }
        """

        current_tracks = []

        # No faces detected
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["missed"] += 1

                if self.tracks[track_id]["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]

            return current_tracks

        detection_centers = [
            self._get_center(bbox)
            for bbox in detections
        ]

        # If there are no existing tracks,
        # create a new track for every detected face.
        if len(self.tracks) == 0:

            for bbox, center in zip(detections, detection_centers):

                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[track_id] = {
                    "bbox": bbox,
                    "center": center,
                    "missed": 0
                }

                current_tracks.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "center": center
                })

            return current_tracks

        # Keep track of which existing tracks have already
        # been assigned to a detection.
        matched_tracks = set()

        for bbox, center in zip(detections, detection_centers):

            best_track_id = None
            best_distance = float("inf")

            # Find the closest existing track.
            for track_id, track in self.tracks.items():

                if track_id in matched_tracks:
                    continue

                distance = self._calculate_distance(
                    center,
                    track["center"]
                )

                if (
                    distance < best_distance
                    and distance <= self.max_distance
                ):
                    best_distance = distance
                    best_track_id = track_id

            # Existing track matched
            if best_track_id is not None:

                self.tracks[best_track_id]["bbox"] = bbox
                self.tracks[best_track_id]["center"] = center
                self.tracks[best_track_id]["missed"] = 0

                matched_tracks.add(best_track_id)

                current_tracks.append({
                    "track_id": best_track_id,
                    "bbox": bbox,
                    "center": center
                })

            # No existing track matched
            else:

                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[track_id] = {
                    "bbox": bbox,
                    "center": center,
                    "missed": 0
                }

                matched_tracks.add(track_id)

                current_tracks.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "center": center
                })

        # Increase missed count for tracks that were
        # not detected in the current frame.
        for track_id in list(self.tracks.keys()):

            if track_id not in matched_tracks:

                self.tracks[track_id]["missed"] += 1

                if self.tracks[track_id]["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]

        return current_tracks