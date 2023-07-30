from typing import Union

class ProgressListener:
    def on_progress(self, current: Union[int, float], total: Union[int, float], desc: str = None):
        self.total = total

    def on_finished(self, desc: str = None):
        pass