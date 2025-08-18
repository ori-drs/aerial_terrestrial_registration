from PyQt5.QtCore import QObject, pyqtSignal


class PipelineWorker(QObject):
    finished = pyqtSignal(bytes)
    progress = pyqtSignal(str)

    def __init__(self, registration):
        super().__init__()
        self.registration = registration

    def run(self):
        try:
            print("Starting registration...")
            # success = self.registration.registration()

        except Exception as e:
            print(f"Registration failed: {e}")
            pass
            # self.failed.emit(str(e))
