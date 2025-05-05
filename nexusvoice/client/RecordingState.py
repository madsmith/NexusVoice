import threading


class RecordingState:
    STOPPED = 0
    RECORDING = 1
    PENDING = 2

    def __init__(self):
        self.state = RecordingState.STOPPED
        self._lock = threading.Lock()

    def is_recording(self):
        with self._lock:
            return self.state != RecordingState.STOPPED

    def is_confirmed(self):
        with self._lock:
            return self.state == RecordingState.RECORDING

    def start(self):
        with self._lock:
            self.state = RecordingState.PENDING

    def stop(self):
        with self._lock:
            self.state = RecordingState.STOPPED

    def confirm(self):
        with self._lock:
            self.state = RecordingState.RECORDING

    def __str__(self):
        labels = {RecordingState.STOPPED: "STOPPED", RecordingState.RECORDING: "RECORDING", RecordingState.PENDING: "PENDING"}
        return labels[self.state]