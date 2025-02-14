from typing import List
from src.diarization.diarization import Diarization, DiarizationEntry
from src.modelCache import GLOBAL_MODEL_CACHE, ModelCache
from src.vadParallel import ParallelContext

class DiarizationContainer:
    def __init__(self, auth_token: str = None, enable_daemon_process: bool = True, auto_cleanup_timeout_seconds=60, cache: ModelCache = None, diarization_version=None):
        self.auth_token = auth_token
        self.enable_daemon_process = enable_daemon_process
        self.auto_cleanup_timeout_seconds = auto_cleanup_timeout_seconds
        self.diarization_context: ParallelContext = None
        self.cache = cache
        self.model = None
        self.diarization_version = diarization_version

    def run(self, audio_file, **kwargs):
        # Create parallel context if needed
        if self.diarization_context is None and self.enable_daemon_process:
            # Number of processes is set to 1 as we mainly use this in order to clean up GPU memory
            self.diarization_context = ParallelContext(num_processes=1, auto_cleanup_timeout_seconds=self.auto_cleanup_timeout_seconds)
            print("Created diarization context with auto cleanup timeout of %d seconds" % self.auto_cleanup_timeout_seconds)
        
        # Run directly 
        if self.diarization_context is None:
            return self.execute(audio_file, **kwargs)

        # Otherwise run in a separate process
        pool = self.diarization_context.get_pool()

        try:
            result = pool.apply(self.execute, (audio_file,), kwargs)
            return result
        finally:
            self.diarization_context.return_pool(pool)

    def mark_speakers(self, diarization_result: List[DiarizationEntry], whisper_result: dict):
        if self.model is not None:
            return self.model.mark_speakers(diarization_result, whisper_result)

        # Create a new diarization model (calling mark_speakers will not initialize pyannote.audio)
        model = Diarization(self.auth_token, self.diarization_version)
        return model.mark_speakers(diarization_result, whisper_result)

    def get_model(self):
        # Lazy load the model
        if (self.model is None):
            if self.cache:
                print(f"Loading {self.diarization_version} model from cache")
                self.model = self.cache.get(self.diarization_version, lambda : Diarization(self.auth_token, self.diarization_version))
            else:
                print(f"Loading {self.diarization_version} model")
                self.model = Diarization(self.auth_token, self.diarization_version)
        return self.model

    def execute(self, audio_file, **kwargs):
        model = self.get_model()

        # We must use list() here to force the iterator to run, as generators are not picklable
        result = list(model.run(audio_file, **kwargs))
        return result
    
    def cleanup(self):
        if self.diarization_context is not None:
            self.diarization_context.close()

    def __getstate__(self):
        return {
            "auth_token": self.auth_token,
            "enable_daemon_process": self.enable_daemon_process,
            "auto_cleanup_timeout_seconds": self.auto_cleanup_timeout_seconds,
            "diarization_version": self.diarization_version
        }
    
    def __setstate__(self, state):
        self.auth_token = state["auth_token"]
        self.enable_daemon_process = state["enable_daemon_process"]
        self.auto_cleanup_timeout_seconds = state["auto_cleanup_timeout_seconds"]
        self.diarization_context = None
        self.diarization_version = state["diarization_version"]
        self.cache = GLOBAL_MODEL_CACHE
        self.model = None