class SelectiveRecognition:
    def __init__(self, recognition_interval=10):
        self.recognition_interval = recognition_interval
        self.track_frames = {}

    def should_recognize(self, track_id):
        if track_id not in self.track_frames:
            self.track_frames[track_id] = 0
            return True

        self.track_frames[track_id] += 1

        if self.track_frames[track_id] >= self.recognition_interval:
            self.track_frames[track_id] = 0
            return True

        return False

    def remove_track(self, track_id):
        self.track_frames.pop(track_id, None)