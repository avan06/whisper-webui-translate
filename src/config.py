from enum import Enum

import os
from typing import List, Dict, Literal


class ModelConfig:
    def __init__(self, name: str, url: str, path: str = None, type: str = "whisper", tokenizer_url: str = None, revision: str = None, model_file: str = None,):
        """
        Initialize a model configuration.

        name: Name of the model
        url: URL to download the model from
        path: Path to the model file. If not set, the model will be downloaded from the URL.
        type: Type of model. Can be whisper or huggingface.
        revision: [by transformers] The specific model version to use.
            It can be a branch name, a tag name, or a commit id, 
            since we use a git-based system for storing models and other artifacts on huggingface.co, 
            so revision can be any identifier allowed by git.
        model_file: The name of the model file in repo or directory.[from marella/ctransformers] 
        """
        self.name = name
        self.url = url
        self.path = path
        self.type = type
        self.tokenizer_url = tokenizer_url
        self.revision = revision
        self.model_file = model_file

VAD_INITIAL_PROMPT_MODE_VALUES=["prepend_all_segments", "prepend_first_segment", "json_prompt_mode"]

class VadInitialPromptMode(Enum):
    PREPEND_ALL_SEGMENTS = 1
    PREPREND_FIRST_SEGMENT = 2
    JSON_PROMPT_MODE = 3

    @staticmethod
    def from_string(s: str):
        normalized = s.lower() if s is not None and len(s) > 0 else None

        if normalized == "prepend_all_segments":
            return VadInitialPromptMode.PREPEND_ALL_SEGMENTS
        elif normalized == "prepend_first_segment":
            return VadInitialPromptMode.PREPREND_FIRST_SEGMENT
        elif normalized == "json_prompt_mode":
            return VadInitialPromptMode.JSON_PROMPT_MODE
        elif normalized is not None and normalized != "":
            raise ValueError(f"Invalid value for VadInitialPromptMode: {s}")
        else:
            return None

class ApplicationConfig:
    def __init__(self, models: Dict[Literal["whisper", "m2m100", "nllb", "mt5", "ALMA", "madlad400"], List[ModelConfig]],
                 input_audio_max_duration: int = 600, share: bool = False, server_name: str = None, server_port: int = 7860, 
                 queue_concurrency_count: int = 1, delete_uploaded_files: bool = True,
                 whisper_implementation: str = "whisper", default_model_name: str = "medium", 
                 default_vad: str = "silero-vad", 
                 vad_parallel_devices: str = "", vad_cpu_cores: int = 1, vad_process_timeout: int = 1800, 
                 auto_parallel: bool = False, output_dir: str = None,
                 model_dir: str = None, device: str = None, 
                 verbose: bool = True, task: str = "transcribe", language: str = None,
                 vad_initial_prompt_mode: str = "prepend_first_segment ", 
                 vad_merge_window: float = 5, vad_max_merge_size: float = 30,
                 vad_padding: float = 1, vad_prompt_window: float = 3,
                 temperature: float = 0, best_of: int = 5, beam_size: int = 5,
                 patience: float = None, length_penalty: float = None,
                 suppress_tokens: str = "-1", initial_prompt: str = None,
                 condition_on_previous_text: bool = True, fp16: bool = True,
                 compute_type: str = "float16", 
                 temperature_increment_on_fallback: float = 0.2, compression_ratio_threshold: float = 2.4,
                 logprob_threshold: float = -1.0, no_speech_threshold: float = 0.6,
                 repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
                 # Word timestamp settings
                 word_timestamps: bool = True, prepend_punctuations: str = "\"\'“¿([{-",
                 append_punctuations: str = "\"\'.。,，!！?？:：”)]}、", 
                 highlight_words: bool = False,
                 # Diarization
                 auth_token: str = None, diarization: bool = False, diarization_speakers: int = 2,
                 diarization_min_speakers: int = 1, diarization_max_speakers: int = 5,
                 diarization_process_timeout: int = 60,
                 # Translation
                 translation_batch_size: int = 2,
                 translation_no_repeat_ngram_size: int = 3,
                 translation_num_beams: int = 2,
                 translation_torch_dtype_float16: bool = True,
                 # Whisper Segments Filter
                 whisper_segments_filter: bool = False,
                 whisper_segments_filters: List[str] = [],
                 ):
        
        self.models = models
        
        # WebUI settings
        self.input_audio_max_duration = input_audio_max_duration
        self.share = share
        self.server_name = server_name
        self.server_port = server_port
        self.queue_concurrency_count = queue_concurrency_count
        self.delete_uploaded_files = delete_uploaded_files

        self.whisper_implementation = whisper_implementation
        self.default_model_name = default_model_name
        self.default_vad = default_vad
        self.vad_parallel_devices = vad_parallel_devices
        self.vad_cpu_cores = vad_cpu_cores
        self.vad_process_timeout = vad_process_timeout
        self.auto_parallel = auto_parallel
        self.output_dir = output_dir

        self.model_dir = model_dir
        self.device = device
        self.verbose = verbose
        self.task = task
        self.language = language
        self.vad_initial_prompt_mode = vad_initial_prompt_mode
        self.vad_merge_window = vad_merge_window
        self.vad_max_merge_size = vad_max_merge_size
        self.vad_padding = vad_padding
        self.vad_prompt_window = vad_prompt_window
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.suppress_tokens = suppress_tokens
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.fp16 = fp16
        self.compute_type = compute_type
        self.temperature_increment_on_fallback = temperature_increment_on_fallback
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        # Word timestamp settings
        self.word_timestamps = word_timestamps
        self.prepend_punctuations = prepend_punctuations
        self.append_punctuations = append_punctuations
        self.highlight_words = highlight_words
        
        # Diarization settings
        self.auth_token = auth_token
        self.diarization = diarization
        self.diarization_speakers = diarization_speakers
        self.diarization_min_speakers = diarization_min_speakers
        self.diarization_max_speakers = diarization_max_speakers
        self.diarization_process_timeout = diarization_process_timeout
        # Translation
        self.translation_batch_size = translation_batch_size
        self.translation_no_repeat_ngram_size = translation_no_repeat_ngram_size
        self.translation_num_beams = translation_num_beams
        self.translation_torch_dtype_float16 = translation_torch_dtype_float16
        # Whisper Segments Filter
        self.whisper_segments_filter = whisper_segments_filter
        self.whisper_segments_filters = whisper_segments_filters

    def get_model_names(self, name: str):
        return [ x.name for x in self.models[name] ]

    def update(self, **new_values):
        result = ApplicationConfig(**self.__dict__)

        for key, value in new_values.items():
            setattr(result, key, value)
        return result

    @staticmethod
    def create_default(**kwargs):
        app_config = ApplicationConfig.parse_file(os.environ.get("WHISPER_WEBUI_CONFIG", "config.json5"))

        # Update with kwargs
        if len(kwargs) > 0:
            app_config = app_config.update(**kwargs)
        return app_config

    @staticmethod
    def parse_file(config_path: str):
        import json5

        with open(config_path, "r", encoding="utf-8") as f:
            # Load using json5
            data = json5.load(f)
            data_models = data.pop("models", [])
            models: Dict[Literal["whisper", "m2m100", "nllb", "mt5", "ALMA", "madlad400"], List[ModelConfig]] = {
                key: [ModelConfig(**item) for item in value]
                for key, value in data_models.items()
            }

            return ApplicationConfig(models, **data)
