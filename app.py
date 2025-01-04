from datetime import datetime
import json
import math
from typing import Iterator, Union, List, Dict, Any
import argparse

from io import StringIO
import time
import os
import pathlib
import tempfile
import zipfile
import numpy as np

import torch

from src.config import VAD_INITIAL_PROMPT_MODE_VALUES, ApplicationConfig, VadInitialPromptMode
from src.diarization.diarization import Diarization
from src.diarization.diarizationContainer import DiarizationContainer
from src.hooks.progressListener import ProgressListener
from src.hooks.subTaskProgressListener import SubTaskProgressListener
from src.hooks.whisperProgressHook import create_progress_listener_handle
from src.modelCache import ModelCache
from src.prompts.jsonPromptStrategy import JsonPromptStrategy
from src.prompts.prependPromptStrategy import PrependPromptStrategy
from src.source import get_audio_source_collection
from src.vadParallel import ParallelContext, ParallelTranscription

# External programs
import ffmpeg

# UI
import gradio as gr

from src.download import ExceededMaximumDuration, download_url
from src.utils import optional_int, slugify, str2bool, write_srt, write_srt_original, write_vtt
from src.vad import AbstractTranscription, NonSpeechStrategy, PeriodicTranscriptionConfig, TranscriptionConfig, VadPeriodicTranscription, VadSileroTranscription
from src.whisper.abstractWhisperContainer import AbstractWhisperContainer
from src.whisper.whisperFactory import create_whisper_container
from src.translation.translationModel import TranslationModel
from src.translation.translationLangs import (TranslationLang,
                                              _TO_LANG_CODE_WHISPER, sort_lang_by_whisper_codes,
                                              get_lang_from_whisper_name, get_lang_from_whisper_code, get_lang_from_nllb_name, get_lang_from_m2m100_name, get_lang_from_seamlessT_Tx_name, 
                                              get_lang_whisper_names, get_lang_nllb_names, get_lang_m2m100_names, get_lang_seamlessT_Tx_names)
import re
import shutil
import zhconv
import tqdm
import traceback

# Configure more application defaults in config.json5

# Gradio seems to truncate files without keeping the extension, so we need to truncate the file prefix ourself 
MAX_FILE_PREFIX_LENGTH = 17

# Limit auto_parallel to a certain number of CPUs (specify vad_cpu_cores to get a higher number)
MAX_AUTO_CPU_CORES = 8

class VadOptions:
    def __init__(self, vad: str = None, vadMergeWindow: float = 5, vadMaxMergeSize: float = 150, vadPadding: float = 1, vadPromptWindow: float = 1, 
                                        vadInitialPromptMode: Union[VadInitialPromptMode, str] = VadInitialPromptMode.PREPREND_FIRST_SEGMENT):
        self.vad = vad
        self.vadMergeWindow = vadMergeWindow
        self.vadMaxMergeSize = vadMaxMergeSize
        self.vadPadding = vadPadding
        self.vadPromptWindow = vadPromptWindow
        self.vadInitialPromptMode = vadInitialPromptMode if isinstance(vadInitialPromptMode, VadInitialPromptMode) \
                                        else VadInitialPromptMode.from_string(vadInitialPromptMode)

class WhisperTranscriber:
    def __init__(self, input_audio_max_duration: float = None, vad_process_timeout: float = None, 
                 vad_cpu_cores: int = 1, delete_uploaded_files: bool = False, output_dir: str = None, 
                 app_config: ApplicationConfig = None):
        self.model_cache = ModelCache()
        self.parallel_device_list = None
        self.gpu_parallel_context = None
        self.cpu_parallel_context = None
        self.vad_process_timeout = vad_process_timeout
        self.vad_cpu_cores = vad_cpu_cores

        self.vad_model = None
        self.inputAudioMaxDuration = input_audio_max_duration
        self.deleteUploadedFiles = delete_uploaded_files
        self.output_dir = output_dir

        # Support for diarization
        self.diarization: DiarizationContainer = None
        # Dictionary with parameters to pass to diarization.run - if None, diarization is not enabled
        self.diarization_kwargs = None
        self.app_config = app_config

    def set_parallel_devices(self, vad_parallel_devices: str):
        self.parallel_device_list = [ device.strip() for device in vad_parallel_devices.split(",") ] if vad_parallel_devices else None

    def set_auto_parallel(self, auto_parallel: bool):
        if auto_parallel:
            if torch.cuda.is_available():
                self.parallel_device_list = [ str(gpu_id) for gpu_id in range(torch.cuda.device_count())]

            self.vad_cpu_cores = min(os.cpu_count(), MAX_AUTO_CPU_CORES)
            print("[Auto parallel] Using GPU devices " + str(self.parallel_device_list) + " and " + str(self.vad_cpu_cores) + " CPU cores for VAD/transcription.")

    def set_diarization(self, auth_token: str, enable_daemon_process: bool = True, **kwargs):
        if self.diarization is None:
            self.diarization = DiarizationContainer(auth_token=auth_token, enable_daemon_process=enable_daemon_process, 
                                                    auto_cleanup_timeout_seconds=self.app_config.diarization_process_timeout, 
                                                    cache=self.model_cache)
        # Set parameters
        self.diarization_kwargs = kwargs

    def unset_diarization(self):
        if self.diarization is not None:
            self.diarization.cleanup()
        self.diarization_kwargs = None

    # Entry function for the simple or full tab, Queue mode disabled: progress bars will not be shown
    def transcribe_entry(self, data: dict): return self.transcribe_entry_progress(data)

    # Entry function for the simple or full tab with progress, Progress tracking requires queuing to be enabled
    def transcribe_entry_progress(self, data: dict, progress=gr.Progress()):
        dataDict = {}
        for key, value in data.items():
            dataDict.update({key.elem_id: value})
            
        return self.transcribe_webui(dataDict, progress=progress)
    
    def transcribe_webui(self, decodeOptions: dict, progress: gr.Progress = None):
        """
        Transcribe an audio file using Whisper
        https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L37
        Parameters
        ----------
        model: Whisper
            The Whisper model instance

        temperature: Union[float, Tuple[float, ...]]
            Temperature for sampling. It can be a tuple of temperatures, which will be successively used
            upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

        compression_ratio_threshold: float
            If the gzip compression ratio is above this value, treat as failed

        logprob_threshold: float
            If the average log probability over sampled tokens is below this value, treat as failed

        no_speech_threshold: float
            If the no_speech probability is higher than this value AND the average log probability
            over sampled tokens is below `logprob_threshold`, consider the segment as silent

        condition_on_previous_text: bool
            if True, the previous output of the model is provided as a prompt for the next window;
            disabling may make the text inconsistent across windows, but the model becomes less prone to
            getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        word_timestamps: bool
            Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
            and include the timestamps for each word in each segment.

        prepend_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with the next word

        append_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with the previous word

        initial_prompt: Optional[str]
            Optional text to provide as a prompt for the first window. This can be used to provide, or
            "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
            to make it more likely to predict those word correctly.

        decode_options: dict
            Keyword arguments to construct `DecodingOptions` instances
            https://github.com/openai/whisper/blob/main/whisper/decoding.py#L81
            
            task: str = "transcribe"
                whether to perform X->X "transcribe" or X->English "translate"

            language: Optional[str] = None
                language that the audio is in; uses detected language if None

            temperature: float = 0.0
            sample_len: Optional[int] = None  # maximum number of tokens to sample
            best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
            beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
            patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)
                sampling-related options

            length_penalty: Optional[float] = None
                "alpha" in Google NMT, or None for length norm, when ranking generations
                to select which to return among the beams or best-of-N samples

            prompt: Optional[Union[str, List[int]]] = None  # for the previous context
            prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context
                text or tokens to feed as the prompt or the prefix; for more info:
                https://github.com/openai/whisper/discussions/117#discussioncomment-3727051

            suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
            suppress_blank: bool = True  # this will suppress blank outputs
                list of tokens ids (or comma-separated token ids) to suppress
                "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`

            without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
            max_initial_timestamp: Optional[float] = 1.0
                timestamp sampling options

            fp16: bool = True  # use fp16 for most of the calculation
                implementation details
        repetition_penalty: float
            The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.
        no_repeat_ngram_size: int
            The model ensures that a sequence of words of no_repeat_ngram_size isn’t repeated in the output sequence. If specified, it must be a positive integer greater than 1.
        """
        try:
            whisperModelName: str = decodeOptions.pop("whisperModelName")
            whisperLangName:  str = decodeOptions.pop("whisperLangName")

            sourceInput:    str  = decodeOptions.pop("sourceInput")
            urlData:        str  = decodeOptions.pop("urlData")
            multipleFiles:  List = decodeOptions.pop("multipleFiles")
            microphoneData: str  = decodeOptions.pop("microphoneData")
            task:           str  = decodeOptions.pop("task")
            
            vad:                      str   = decodeOptions.pop("vad")
            vadMergeWindow:           float = decodeOptions.pop("vadMergeWindow")
            vadMaxMergeSize:          float = decodeOptions.pop("vadMaxMergeSize")
            vadPadding:               float = decodeOptions.pop("vadPadding", self.app_config.vad_padding)
            vadPromptWindow:          float = decodeOptions.pop("vadPromptWindow", self.app_config.vad_prompt_window)
            vadInitialPromptMode:     str   = decodeOptions.pop("vadInitialPromptMode", self.app_config.vad_initial_prompt_mode)
            self.vad_process_timeout: float = decodeOptions.pop("vadPocessTimeout", self.vad_process_timeout)
            
            self.whisperSegmentsFilters: List[List] = []
            inputFilter: bool = decodeOptions.pop("whisperSegmentsFilter", None)
            inputFilters = []
            for idx in range(1,len(self.app_config.whisper_segments_filters) + 1,1):
                inputFilters.append(decodeOptions.pop(f"whisperSegmentsFilter{idx}", None))
            inputFilters = filter(None, inputFilters)
            if inputFilter:
                for inputFilter in inputFilters:
                    self.whisperSegmentsFilters.append([])
                    self.whisperSegmentsFilters[-1].append(inputFilter)
                    for text in inputFilter.split(","):
                        result = []
                        subFilter = [text] if "||" not in text else [strFilter_ for strFilter_ in text.lstrip("(").rstrip(")").split("||") if strFilter_]
                        for string in subFilter:
                            conditions = [condition for condition in string.split(" ") if condition]
                            if len(conditions) == 1 and conditions[0] == "segment_last":
                                pass
                            elif len(conditions) == 3:
                                conditions[-1] = float(conditions[-1])
                            else:
                                continue
                            result.append(conditions)
                        self.whisperSegmentsFilters[-1].append(result)

            diarization:              bool = decodeOptions.pop("diarization", False)
            diarization_speakers:     int  = decodeOptions.pop("diarization_speakers", 2)
            diarization_min_speakers: int  = decodeOptions.pop("diarization_min_speakers", 1)
            diarization_max_speakers: int  = decodeOptions.pop("diarization_max_speakers", 8)
            highlight_words:          bool = decodeOptions.pop("highlight_words", False)
            
            temperature: float = decodeOptions.pop("temperature", None)
            temperature_increment_on_fallback: float = decodeOptions.pop("temperature_increment_on_fallback", None)
            
            whisperRepetitionPenalty: float = decodeOptions.get("repetition_penalty", None)
            whisperNoRepeatNgramSize: int = decodeOptions.get("no_repeat_ngram_size", None)
            if whisperRepetitionPenalty is not None and whisperRepetitionPenalty <= 1.0:
                decodeOptions.pop("repetition_penalty")
            if whisperNoRepeatNgramSize is not None and whisperNoRepeatNgramSize <= 1:
                decodeOptions.pop("no_repeat_ngram_size")

            for key, value in list(decodeOptions.items()):
                if value == "":
                    del decodeOptions[key]

            # word_timestamps             = decodeOptions.get("word_timestamps", False)
            # condition_on_previous_text  = decodeOptions.get("condition_on_previous_text", False)
            # prepend_punctuations        = decodeOptions.get("prepend_punctuations", None)
            # append_punctuations         = decodeOptions.get("append_punctuations", None)
            # initial_prompt              = decodeOptions.get("initial_prompt", None)
            # best_of                     = decodeOptions.get("best_of", None)
            # beam_size                   = decodeOptions.get("beam_size", None)
            # patience                    = decodeOptions.get("patience", None)
            # length_penalty              = decodeOptions.get("length_penalty", None)
            # suppress_tokens             = decodeOptions.get("suppress_tokens", None)
            # compression_ratio_threshold = decodeOptions.get("compression_ratio_threshold", None)
            # logprob_threshold           = decodeOptions.get("logprob_threshold", None)

            vadOptions = VadOptions(vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow, vadInitialPromptMode)

            if diarization:
                if diarization_speakers is not None and diarization_speakers < 1:
                    self.set_diarization(auth_token=self.app_config.auth_token, min_speakers=diarization_min_speakers, max_speakers=diarization_max_speakers)
                else:
                    self.set_diarization(auth_token=self.app_config.auth_token, num_speakers=diarization_speakers, min_speakers=diarization_min_speakers, max_speakers=diarization_max_speakers)
            else:
                self.unset_diarization()
                
            # Handle temperature_increment_on_fallback
            if temperature is not None:
                if temperature_increment_on_fallback is not None:
                    temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
                else:
                    temperature = [temperature]
                decodeOptions["temperature"] = temperature

            progress(0, desc="init audio sources")
            
            if sourceInput == "urlData":
                sources = self.__get_source(urlData, None, None)
            elif sourceInput == "multipleFiles":
                sources = self.__get_source(None, multipleFiles, None)
            elif sourceInput == "microphoneData":
                sources = self.__get_source(None, None, microphoneData)
                
            if (len(sources) == 0):
                raise Exception("init audio sources failed...")
            
            try:
                progress(0, desc="init whisper model")
                whisperLang: TranslationLang = get_lang_from_whisper_name(whisperLangName)
                whisperLangCode = whisperLang.whisper.code if whisperLang is not None and whisperLang.whisper is not None else None
                selectedModel = whisperModelName if whisperModelName is not None else "base"

                model = create_whisper_container(whisper_implementation=self.app_config.whisper_implementation, 
                                                 model_name=selectedModel, compute_type=self.app_config.compute_type, 
                                                 cache=self.model_cache, models=self.app_config.models["whisper"])
                
                progress(0, desc="init translate model")
                translationLang, translationModel = self.initTranslationModel(whisperLangName, whisperLang, decodeOptions)

                progress(0, desc="init transcribe")
                # Result
                download = []
                zip_file_lookup = {}
                text = ""
                vtt = ""
                filterLogs = ""

                # Write result
                downloadDirectory = tempfile.mkdtemp()
                source_index = 0
                extra_tasks_count = 1 if translationLang is not None else 0

                outputDirectory = self.output_dir if self.output_dir is not None else downloadDirectory

                # Progress
                total_duration = sum([source.get_audio_duration() for source in sources])
                current_progress = 0

                # A listener that will report progress to Gradio
                root_progress_listener = self._create_progress_listener(progress)
                sub_task_total = 1/(len(sources)+extra_tasks_count*len(sources))

                # Execute whisper
                for idx, source in enumerate(sources):
                    source_prefix = ""
                    source_audio_duration = source.get_audio_duration()

                    if (len(sources) > 1):
                        # Prefix (minimum 2 digits)
                        source_index += 1
                        source_prefix = str(source_index).zfill(2) + "_"
                        print("Transcribing ", source.source_path)

                    scaled_progress_listener = SubTaskProgressListener(root_progress_listener, 
                                                   base_task_total=1,
                                                   sub_task_start=idx*1/len(sources),
                                                   sub_task_total=sub_task_total)

                    # Transcribe
                    result = self.transcribe_file(model, source.source_path, whisperLangCode, task, vadOptions, scaled_progress_listener, **decodeOptions)
                    filterLog = result.get("filterLog", None)
                    if filterLog:
                        filterLogs += source.get_full_name() + ":\n" + filterLog + "\n\n"

                    if translationModel is not None and whisperLang is None and result["language"] is not None and len(result["language"]) > 0:
                        whisperLang = get_lang_from_whisper_code(result["language"])
                        translationModel.whisperLang = whisperLang
                        
                    short_name, suffix = source.get_short_name_suffix(max_length=self.app_config.input_max_file_name_length)
                    filePrefix = slugify(source_prefix + short_name, allow_unicode=True)

                    # Update progress
                    current_progress += source_audio_duration

                    source_download, source_text, source_vtt = self.write_result(result, whisperLang, translationModel, filePrefix + suffix.replace(".", "_"), outputDirectory, highlight_words, scaled_progress_listener)

                    if self.app_config.merge_subtitle_with_sources and self.app_config.output_dir is not None:
                        print("\nmerge subtitle(srt) with source file [" + source.source_name + "]\n")
                        outRsult = ""
                        try:
                            srt_path = source_download[0]
                            save_path = os.path.join(self.app_config.output_dir, filePrefix)
                            # save_without_ext, ext = os.path.splitext(save_path)
                            source_lang = "." + whisperLang.whisper.code if whisperLang is not None and whisperLang.whisper is not None else ""
                            translate_lang = "." + translationLang.nllb.code if translationLang is not None else ""
                            output_with_srt = save_path + source_lang + translate_lang + suffix
        
                            #ffmpeg -i "input.mp4" -i "input.srt" -c copy -c:s mov_text output.mp4
                            input_file = ffmpeg.input(source.source_path)
                            input_srt = ffmpeg.input(srt_path)
                            out = ffmpeg.output(input_file, input_srt, output_with_srt, vcodec='copy', acodec='copy', scodec='mov_text')
                            outRsult = out.run(overwrite_output=True)
                        except Exception as e:
                            print(traceback.format_exc())
                            print("Error merge subtitle with source file: \n" + source.source_path + ", \n" + str(e), outRsult)
                    elif self.app_config.save_downloaded_files and self.app_config.output_dir is not None and urlData:
                        print("Saving downloaded file [" + source.source_name + "]")
                        try:
                            save_path = os.path.join(self.app_config.output_dir, filePrefix)
                            shutil.copy(source.source_path, save_path + suffix)
                        except Exception as e:
                            print(traceback.format_exc())
                            print("Error saving downloaded file: \n" + source.source_path + ", \n" + str(e))

                    if len(sources) > 1:
                        # Add new line separators
                        if (len(source_text) > 0):
                            source_text += os.linesep + os.linesep
                        if (len(source_vtt) > 0):
                            source_vtt += os.linesep + os.linesep

                        # Append file name to source text too
                        source_text = source.get_full_name() + ":" + os.linesep + source_text
                        source_vtt = source.get_full_name() + ":" + os.linesep + source_vtt

                    # Add to result
                    download.extend(source_download)
                    text += source_text
                    vtt += source_vtt

                    if (len(sources) > 1):
                        # Zip files support at least 260 characters, but we'll play it safe and use 200
                        zipFilePrefix = slugify(source_prefix + source.get_short_name(max_length=200), allow_unicode=True)

                        # File names in ZIP file can be longer
                        for source_download_file in source_download:
                            # Get file postfix (after last -)
                            filePostfix = os.path.basename(source_download_file).split("-")[-1]
                            zip_file_name = zipFilePrefix + "-" + filePostfix
                            zip_file_lookup[source_download_file] = zip_file_name

                # Create zip file from all sources
                if len(sources) > 1:
                    downloadAllPath = os.path.join(downloadDirectory, "All_Output-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip")

                    with zipfile.ZipFile(downloadAllPath, 'w', zipfile.ZIP_DEFLATED) as zip:
                        for download_file in download:
                            # Get file name from lookup
                            zip_file_name = zip_file_lookup.get(download_file, os.path.basename(download_file))
                            zip.write(download_file, arcname=zip_file_name)

                    download.insert(0, downloadAllPath)
                    
                filterLogText = [gr.Text(visible=False)] #[Gradio 5.x] AttributeError: type object 'Textbox' has no attribute 'update'
                if filterLogs:
                    filterLogText[0].visible = True
                    filterLogText[0].value = filterLogs
                
                return [download, text, vtt] + filterLogText

            finally:
                # Cleanup source
                if self.deleteUploadedFiles and sourceInput == "urlData":
                    for source in sources:
                        print("Deleting temporary source file: " + source.source_path)
                        try:
                            os.remove(source.source_path)
                        except Exception as e:
                            print(traceback.format_exc())
                            print("Error deleting temporary source file: \n" + source.source_path + ", \n" + str(e))
        
        except ExceededMaximumDuration as e:
            return [], "[ERROR]: Maximum remote video length is " + str(e.maxDuration) + "s, file was " + str(e.videoDuration) + "s", "[ERROR]", ""
        except Exception as e:
            print(traceback.format_exc())
            return [], "Error occurred during transcribe: " + str(e), traceback.format_exc(), ""
        

    def transcribe_file(self, model: AbstractWhisperContainer, audio_path: str, languageCode: str, task: str = None, 
                        vadOptions: VadOptions = VadOptions(), 
                        progressListener: ProgressListener = None, **decodeOptions: dict):
        
        initial_prompt = decodeOptions.pop('initial_prompt', None)

        if progressListener is None:
            # Default progress listener
            progressListener = ProgressListener()

        if ('task' in decodeOptions):
            task = decodeOptions.pop('task')

        initial_prompt_mode = vadOptions.vadInitialPromptMode

        # Set default initial prompt mode
        if (initial_prompt_mode is None):
            initial_prompt_mode = VadInitialPromptMode.PREPREND_FIRST_SEGMENT

        if (initial_prompt_mode == VadInitialPromptMode.PREPEND_ALL_SEGMENTS or 
            initial_prompt_mode == VadInitialPromptMode.PREPREND_FIRST_SEGMENT):
            # Prepend initial prompt
            prompt_strategy = PrependPromptStrategy(initial_prompt, initial_prompt_mode)
        elif (vadOptions.vadInitialPromptMode == VadInitialPromptMode.JSON_PROMPT_MODE):
            # Use a JSON format to specify the prompt for each segment
            prompt_strategy = JsonPromptStrategy(initial_prompt)
        else:
            raise ValueError("Invalid vadInitialPromptMode: " + initial_prompt_mode)

        # Callable for processing an audio file
        whisperCallable = model.create_callback(languageCode, task, prompt_strategy=prompt_strategy, **decodeOptions)

        # The results
        if (vadOptions.vad == 'silero-vad'):
            # Silero VAD where non-speech gaps are transcribed
            process_gaps = self._create_silero_config(NonSpeechStrategy.CREATE_SEGMENT, vadOptions)
            result = self.process_vad(audio_path, whisperCallable, self.vad_model, process_gaps, progressListener=progressListener)
        elif (vadOptions.vad == 'silero-vad-skip-gaps'):
            # Silero VAD where non-speech gaps are simply ignored
            skip_gaps = self._create_silero_config(NonSpeechStrategy.SKIP, vadOptions)
            result = self.process_vad(audio_path, whisperCallable, self.vad_model, skip_gaps, progressListener=progressListener)
        elif (vadOptions.vad == 'silero-vad-expand-into-gaps'):
            # Use Silero VAD where speech-segments are expanded into non-speech gaps
            expand_gaps = self._create_silero_config(NonSpeechStrategy.EXPAND_SEGMENT, vadOptions)
            result = self.process_vad(audio_path, whisperCallable, self.vad_model, expand_gaps, progressListener=progressListener)
        elif (vadOptions.vad == 'periodic-vad'):
            # Very simple VAD - mark every 5 minutes as speech. This makes it less likely that Whisper enters an infinite loop, but
            # it may create a break in the middle of a sentence, causing some artifacts.
            periodic_vad = VadPeriodicTranscription()
            period_config = PeriodicTranscriptionConfig(periodic_duration=vadOptions.vadMaxMergeSize, max_prompt_window=vadOptions.vadPromptWindow)
            result = self.process_vad(audio_path, whisperCallable, periodic_vad, period_config, progressListener=progressListener)

        else:
            if (self._has_parallel_devices()):
                # Use a simple period transcription instead, as we need to use the parallel context
                periodic_vad = VadPeriodicTranscription()
                period_config = PeriodicTranscriptionConfig(periodic_duration=math.inf, max_prompt_window=1)

                result = self.process_vad(audio_path, whisperCallable, periodic_vad, period_config, progressListener=progressListener)
            else:
                # Default VAD
                result = whisperCallable.invoke(audio_path, 0, None, None, progress_listener=progressListener)

        if self.whisperSegmentsFilters:
            querySegmentsResult, filterLog = self.filterSegments(result["segments"])     
            result["segments"] = querySegmentsResult
            if filterLog:
                result["filterLog"] = filterLog

        # Diarization
        if self.diarization and self.diarization_kwargs:
            print("Diarizing ", audio_path)
            diarization_result = list(self.diarization.run(audio_path, **self.diarization_kwargs))

            # Print result
            print("Diarization result: ")
            for entry in diarization_result:
                print(f"  start={entry.start:.1f}s stop={entry.end:.1f}s speaker_{entry.speaker}")

            # Add speakers to result
            result = self.diarization.mark_speakers(diarization_result, result)

        return result
    
    def filterSegments(self, querySegments: List[Dict[str, Any]]):
        try:
            if not self.whisperSegmentsFilters: return
            
            filterIdx = 0
            filterLog = []
            querySegmentsResult = querySegments.copy()
            for idx in range(len(querySegmentsResult),0,-1):
                currentID = idx - 1
                querySegment = querySegmentsResult[currentID]
                for segmentsFilter in self.whisperSegmentsFilters:
                    isFilter: bool = True
                    for idx, strFilter in enumerate(segmentsFilter):
                        if not isFilter: break
                        if idx == 0:
                            filterCondition = strFilter
                            continue

                        isFilter = True
                        for subFilter in strFilter:
                            key: str = subFilter[0]
                            
                            if key == "segment_last":
                                isFilter = querySegment.get(key, None)
                                if isFilter: break
                                continue
                            
                            sign: str = subFilter[1]
                            threshold: float = subFilter[2]
                        
                            if key == "durationLen":
                                value = querySegment["end"] - querySegment["start"]
                            elif key == "textLen":
                                value = len(querySegment["text"])
                            else:
                                value = querySegment[key]
                            
                            if sign == "=" or sign == "==":
                                isFilter = value == threshold
                            elif sign == ">":
                                isFilter = value > threshold
                            elif sign == ">=":
                                isFilter = value >= threshold
                            elif sign == "<":
                                isFilter = value < threshold
                            elif sign == "<=":
                                isFilter = value <= threshold
                            else: isFilter = False
                            
                            if isFilter: break
                    if isFilter: break
                if isFilter:
                    filterLog.append(f"\t{querySegment}\n")
                    del querySegmentsResult[currentID]
                    
            if filterLog:
                filterLog = [f"filter{idx:03d} [{filterCondition}]:\n{log}" for idx, log in enumerate(reversed(filterLog))]

            return querySegmentsResult, "\n".join(filterLog)
        except Exception as e:
            print(traceback.format_exc())
            print("Error filter segments: " + str(e))

    def _create_progress_listener(self, progress: gr.Progress):
        if (progress is None):
            # Dummy progress listener
            return ProgressListener()
        
        class ForwardingProgressListener(ProgressListener):
            def __init__(self, progress: gr.Progress):
                self.progress = progress

            def on_progress(self, current: Union[int, float], total: Union[int, float], desc: str = None):
                # From 0 to 1
                self.progress(current / total, desc=desc)

            def on_finished(self, desc: str = None):
                self.progress(1, desc=desc)

        return ForwardingProgressListener(progress)

    def process_vad(self, audio_path, whisperCallable, vadModel: AbstractTranscription, vadConfig: TranscriptionConfig, 
                    progressListener: ProgressListener = None):
        if (not self._has_parallel_devices()):
            # No parallel devices, so just run the VAD and Whisper in sequence
            return vadModel.transcribe(audio_path, whisperCallable, vadConfig, progressListener=progressListener)

        gpu_devices = self.parallel_device_list

        if (gpu_devices is None or len(gpu_devices) == 0):
            # No GPU devices specified, pass the current environment variable to the first GPU process. This may be NULL.
            gpu_devices = [os.environ.get("CUDA_VISIBLE_DEVICES", None)]

        # Create parallel context if needed
        if (self.gpu_parallel_context is None):
            # Create a context wih processes and automatically clear the pool after 1 hour of inactivity
            self.gpu_parallel_context = ParallelContext(num_processes=len(gpu_devices), auto_cleanup_timeout_seconds=self.vad_process_timeout)
        # We also need a CPU context for the VAD
        if (self.cpu_parallel_context is None):
            self.cpu_parallel_context = ParallelContext(num_processes=self.vad_cpu_cores, auto_cleanup_timeout_seconds=self.vad_process_timeout)

        parallel_vad = ParallelTranscription()
        return parallel_vad.transcribe_parallel(transcription=vadModel, audio=audio_path, whisperCallable=whisperCallable,  
                                                config=vadConfig, cpu_device_count=self.vad_cpu_cores, gpu_devices=gpu_devices, 
                                                cpu_parallel_context=self.cpu_parallel_context, gpu_parallel_context=self.gpu_parallel_context, 
                                                progress_listener=progressListener) 

    def _has_parallel_devices(self):
        return (self.parallel_device_list is not None and len(self.parallel_device_list) > 0) or self.vad_cpu_cores > 1

    def _concat_prompt(self, prompt1, prompt2):
        if (prompt1 is None):
            return prompt2
        elif (prompt2 is None):
            return prompt1
        else:
            return prompt1 + " " + prompt2

    def _create_silero_config(self, non_speech_strategy: NonSpeechStrategy, vadOptions: VadOptions):
        # Use Silero VAD 
        if (self.vad_model is None):
            self.vad_model = VadSileroTranscription() #vad_model is snakers4/silero-vad

        config = TranscriptionConfig(non_speech_strategy = non_speech_strategy, 
                max_silent_period=vadOptions.vadMergeWindow, max_merge_size=vadOptions.vadMaxMergeSize, 
                segment_padding_left=vadOptions.vadPadding, segment_padding_right=vadOptions.vadPadding, 
                max_prompt_window=vadOptions.vadPromptWindow)

        return config

    def write_result(self, result: dict, whisperLang: TranslationLang, translationModel: TranslationModel, source_name: str, output_dir: str, highlight_words: bool = False, progressListener: ProgressListener = None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        text = result["text"]
        segments = result["segments"]
        language = result["language"]
        languageMaxLineWidth = 80 #Use east_asian_width to automatically determine the Character Width of the string, replacing the __get_max_line_width function. 80 latin characters should fit on a 1080p/720p screen

        if translationModel is not None and translationModel.translationLang is not None:
            try:
                segments_progress_listener = SubTaskProgressListener(progressListener, 
                                               base_task_total=progressListener.sub_task_total, 
                                               sub_task_start=1, 
                                               sub_task_total=1)
                pbar = tqdm.tqdm(total=len(segments))
                perf_start_time = time.perf_counter()
                translationModel.load_model()
                for idx, segment in enumerate(segments):
                    seg_text = segment["text"]
                    segment["original"] = seg_text
                    segment["text"] = translationModel.translation(seg_text)
                    pbar.update(1)
                    segments_progress_listener.on_progress(idx+1, len(segments), desc=f"Process segments: {idx}/{len(segments)}")

                translationModel.release_vram()

                if highlight_words and segments[0]["words"] is not None:
                    for idx, segment in enumerate(segments):
                        text = segment["text"]
                        words = segment["words"]
                        total_duration = words[-1]['end'] - words[0]['start'] #Calculate the total duration of the entire sentence
                        total_text_length = len(text)

                        # Allocate lengths to each word
                        duration_ratio_lengths = []
                        total_allocated = 0
                        text_idx = 0  # Track the position in the translated string
                        for word in words:
                            # Calculate the duration of each word as a proportion of the total time
                            word_duration = word['end'] - word['start']
                            duration_ratio = word_duration / total_duration
                            duration_ratio_length = int(duration_ratio * total_text_length)
                            duration_ratio_lengths.append(duration_ratio_length)
                            total_allocated += duration_ratio_length

                        # Distribute remaining characters to avoid 0-duration_ratio_length issues
                        remaining_chars = total_text_length - total_allocated
                        for idx in range(remaining_chars):
                            duration_ratio_lengths[idx % len(words)] += 1  # Distribute the remaining chars evenly

                        # Generate translated words based on the calculated lengths
                        text_idx = 0
                        for idx, word in enumerate(words):
                            text_part = text[text_idx:text_idx + duration_ratio_lengths[idx]]
                            word["word"], word["word_original"] = text_part, word["word"]
                            text_idx += duration_ratio_lengths[idx]

                perf_end_time = time.perf_counter()
                # Call the finished callback
                if segments_progress_listener is not None:
                    segments_progress_listener.on_finished(desc=f"Process segments: {idx}/{len(segments)}")

                print("\n\nprocess segments took {} seconds.\n\n".format(perf_end_time - perf_start_time))
            except Exception as e:
                print(traceback.format_exc())
                print("Error process segments: " + str(e))

        print("Max line Character Width " + str(languageMaxLineWidth) + " for language:" + language)
        vtt = self.__get_subs(result["segments"], "vtt", languageMaxLineWidth, highlight_words=highlight_words)
        srt = self.__get_subs(result["segments"], "srt", languageMaxLineWidth, highlight_words=highlight_words)
        json_result = json.dumps(result, indent=4, ensure_ascii=False)
        srt_original = None
        srt_bilingual = None
        if translationModel is not None and translationModel.translationLang is not None:
            srt_original  = self.__get_subs(result["segments"], "srt_original", languageMaxLineWidth, highlight_words=highlight_words)
            srt_bilingual = self.__get_subs(result["segments"], "srt_bilingual", languageMaxLineWidth, highlight_words=highlight_words)

        whisperLangZho: bool = whisperLang is not None and whisperLang.nllb is not None and whisperLang.nllb.code in ["zho_Hant", "zho_Hans", "yue_Hant"]
        translationZho: bool = translationModel is not None and translationModel.translationLang is not None and translationModel.translationLang.nllb is not None and translationModel.translationLang.nllb.code in ["zho_Hant", "zho_Hans", "yue_Hant"]
        if whisperLangZho or translationZho:
            locale = None
            if whisperLangZho:
                if whisperLang.nllb.code == "zho_Hant":
                    locale = "zh-tw"
                elif whisperLang.nllb.code == "zho_Hans":
                    locale = "zh-cn"
                elif whisperLang.nllb.code == "yue_Hant":
                    locale = "zh-hk"
            if translationZho:
                if translationModel.translationLang.nllb.code == "zho_Hant":
                    locale = "zh-tw"
                elif translationModel.translationLang.nllb.code == "zho_Hans":
                    locale = "zh-cn"
                elif translationModel.translationLang.nllb.code == "yue_Hant":
                    locale = "zh-hk"
            if locale is not None:
                vtt = zhconv.convert(vtt, locale)
                srt = zhconv.convert(srt, locale)
                text = zhconv.convert(text, locale)
                json_result = zhconv.convert(json_result, locale)
                if translationModel is not None and translationModel.translationLang is not None:
                    if srt_original is not None and len(srt_original) > 0:
                        srt_original = zhconv.convert(srt_original, locale)
                    if srt_bilingual is not None and len(srt_bilingual) > 0:
                        srt_bilingual = zhconv.convert(srt_bilingual, locale)

        output_files = []
        output_files.append(self.__create_file(srt, output_dir, source_name + "-subs.srt"));
        output_files.append(self.__create_file(vtt, output_dir, source_name + "-subs.vtt"));
        output_files.append(self.__create_file(text, output_dir, source_name + "-transcript.txt"));
        output_files.append(self.__create_file(json_result, output_dir, source_name + "-result.json"));
        if srt_original is not None and len(srt_original) > 0:
            output_files.append(self.__create_file(srt_original, output_dir, source_name + "-original.srt"));
        if srt_bilingual is not None and len(srt_bilingual) > 0:
            output_files.append(self.__create_file(srt_bilingual, output_dir, source_name + "-bilingual.srt"));

        return output_files, text, srt_bilingual if srt_bilingual is not None and len(srt_bilingual) > 0 else vtt

    def clear_cache(self):
        self.model_cache.clear()
        self.vad_model = None

    def __get_source(self, urlData, multipleFiles, microphoneData):
        return get_audio_source_collection(urlData, multipleFiles, microphoneData, self.inputAudioMaxDuration)

    def __get_subs(self, segments: Iterator[dict], format: str, maxLineWidth: int, highlight_words: bool = False) -> str:
        segmentStream = StringIO()

        if format == 'vtt':
            write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth, highlight_words=highlight_words)
        elif format == 'srt':
            write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth, highlight_words=highlight_words)
        elif format == 'srt_original':
            write_srt_original(segments, file=segmentStream, maxLineWidth=maxLineWidth, highlight_words=highlight_words)
        elif format == 'srt_bilingual':
            write_srt_original(segments, file=segmentStream, maxLineWidth=maxLineWidth, highlight_words=highlight_words, bilingual=True)
        else:
            raise Exception("Unknown format " + format)

        segmentStream.seek(0)
        return segmentStream.read()

    def __create_file(self, text: str, directory: str, fileName: str) -> str:
        # Write the text to a file
        with open(os.path.join(directory, fileName), 'w+', encoding="utf-8") as file:
            file.write(text)

        return file.name

    def close(self):
        print("Closing parallel contexts")
        self.clear_cache()

        if (self.gpu_parallel_context is not None):
            self.gpu_parallel_context.close()
        if (self.cpu_parallel_context is not None):
            self.cpu_parallel_context.close()

        # Cleanup diarization
        if (self.diarization is not None):
            self.diarization.cleanup()
            self.diarization = None

    # Entry function for the simple or full tab, Queue mode disabled: progress bars will not be shown
    def translation_entry(self, data: dict): return self.translation_entry_progress(data)

    # Entry function for the simple or full tab with progress, Progress tracking requires queuing to be enabled
    def translation_entry_progress(self, data: dict, progress=gr.Progress()):
        dataDict = {}
        for key, value in data.items():
            dataDict.update({key.elem_id: value})
            
        return self.translation_webui(dataDict, progress=progress)
    
    def translation_webui(self, dataDict: dict, progress: gr.Progress = None):
        try:
            inputText:          str = dataDict.pop("inputText")
            inputLangName:      str = dataDict.pop("inputLangName")
            inputLang: TranslationLang = get_lang_from_whisper_name(inputLangName)
            
            progress(0, desc="init translate model")
            translationLang, translationModel = self.initTranslationModel(inputLangName, inputLang, dataDict)
            
            translationEnbaleBilingual:  bool = dataDict.pop("translationEnbaleBilingual")
            translationDetectLineBreaks: bool = dataDict.pop("translationDetectLineBreaks")

            result = []
            if translationModel and translationModel.translationLang:
                try:
                    inputTexts = inputText.split("\n")

                    progress(0, desc="Translation starting...")
                    
                    perf_start_time = time.perf_counter()
                    translationModel.load_model()
                    
                    def doTranslation(text: str):
                        if translationEnbaleBilingual:
                            result.append(text)
                        result.append(translationModel.translation(text))

                    temporaryText = ""
                    for idx, text in enumerate(tqdm.tqdm(inputTexts)):
                        if not text or re.match("""^[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,\-.\/:;<=>?@\[\]^_`{|}~\d ]+$""", text.strip()):
                            if temporaryText:
                                doTranslation(temporaryText)
                                temporaryText = ""
                            result.append(text)
                        else:
                            if translationDetectLineBreaks and ((not text.rstrip().endswith(".") and not text.rstrip().endswith("。")) or temporaryText):
                                if temporaryText:
                                    temporaryText = temporaryText.rstrip() + " "
                                temporaryText += text
                                continue
                            doTranslation(text)
                        progress((idx+1)/len(inputTexts), desc=f"Process inputText: {idx+1}/{len(inputTexts)}")

                    if temporaryText:
                        doTranslation(temporaryText)

                    translationModel.release_vram()
                    perf_end_time = time.perf_counter()
                    # Call the finished callback
                    progress(1, desc=f"Process inputText: {idx+1}/{len(inputTexts)}")

                    print("\n\nprocess inputText took {} seconds.\n\n".format(perf_end_time - perf_start_time))
                except Exception as e:
                    print(traceback.format_exc())
                    print("Error process inputText: " + str(e))

            resultStr = "\n".join(result)
            
            translationZho: bool = translationModel and translationModel.translationLang and translationModel.translationLang.nllb and translationModel.translationLang.nllb.code in ["zho_Hant", "zho_Hans", "yue_Hant"]
            if translationZho:
                if translationModel.translationLang.nllb.code == "zho_Hant":
                    locale = "zh-tw"
                elif translationModel.translationLang.nllb.code == "zho_Hans":
                    locale = "zh-cn"
                elif translationModel.translationLang.nllb.code == "yue_Hant":
                    locale = "zh-hk"
                resultStr = zhconv.convert(resultStr, locale)
                
            return resultStr
        except Exception as e:
            print(traceback.format_exc())
            return "Error occurred during transcribe: " + str(e) + "\n\n" + traceback.format_exc()
    
    def initTranslationModel(self, inputLangName: str, inputLang: TranslationLang, dataDict: dict):
        translateInput:     str = dataDict.pop("translateInput")
        m2m100ModelName:    str = dataDict.pop("m2m100ModelName")
        m2m100LangName:     str = dataDict.pop("m2m100LangName")
        nllbModelName:      str = dataDict.pop("nllbModelName")
        nllbLangName:       str = dataDict.pop("nllbLangName")
        mt5ModelName:       str = dataDict.pop("mt5ModelName")
        mt5LangName:        str = dataDict.pop("mt5LangName")
        ALMAModelName:      str = dataDict.pop("ALMAModelName")
        ALMALangName:       str = dataDict.pop("ALMALangName")
        madlad400ModelName: str = dataDict.pop("madlad400ModelName")
        madlad400LangName:  str = dataDict.pop("madlad400LangName")
        seamlessModelName:  str = dataDict.pop("seamlessModelName")
        seamlessLangName:   str = dataDict.pop("seamlessLangName")
        LlamaModelName:     str = dataDict.pop("LlamaModelName")
        LlamaLangName:      str = dataDict.pop("LlamaLangName")
        
        translationBatchSize:         int  = dataDict.pop("translationBatchSize")
        translationNoRepeatNgramSize: int  = dataDict.pop("translationNoRepeatNgramSize")
        translationNumBeams:          int  = dataDict.pop("translationNumBeams")
        translationTorchDtypeFloat16: bool = dataDict.pop("translationTorchDtypeFloat16")
        translationUsingBitsandbytes: str  = dataDict.pop("translationUsingBitsandbytes")
        
        translationLang = None
        translationModel = None
        if translateInput == "m2m100" and m2m100LangName is not None and len(m2m100LangName) > 0:
            selectedModelName = m2m100ModelName if m2m100ModelName is not None and len(m2m100ModelName) > 0 else "m2m100_418M/facebook"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["m2m100"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_m2m100_name(m2m100LangName)
        elif translateInput == "nllb" and nllbLangName is not None and len(nllbLangName) > 0:
            selectedModelName = nllbModelName if nllbModelName is not None and len(nllbModelName) > 0 else "nllb-200-distilled-600M/facebook"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["nllb"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_nllb_name(nllbLangName)
        elif translateInput == "mt5" and mt5LangName is not None and len(mt5LangName) > 0:
            selectedModelName = mt5ModelName if mt5ModelName is not None and len(mt5ModelName) > 0 else "mt5-zh-ja-en-trimmed/K024"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["mt5"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_m2m100_name(mt5LangName)
        elif translateInput == "ALMA" and ALMALangName is not None and len(ALMALangName) > 0:
            selectedModelName = ALMAModelName if ALMAModelName is not None and len(ALMAModelName) > 0 else "ALMA-7B-ct2:int8_float16/avan"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["ALMA"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_m2m100_name(ALMALangName)
        elif translateInput == "madlad400" and madlad400LangName is not None and len(madlad400LangName) > 0:
            selectedModelName = madlad400ModelName if madlad400ModelName is not None and len(madlad400ModelName) > 0 else "madlad400-3b-mt-ct2-int8_float16/SoybeanMilk"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["madlad400"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_m2m100_name(madlad400LangName)
        elif translateInput == "seamless" and seamlessLangName is not None and len(seamlessLangName) > 0:
            selectedModelName = seamlessModelName if seamlessModelName is not None and len(seamlessModelName) > 0 else "seamless-m4t-v2-large/facebook"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["seamless"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_seamlessT_Tx_name(seamlessLangName)
        elif translateInput == "Llama" and LlamaLangName is not None and len(LlamaLangName) > 0:
            selectedModelName = LlamaModelName if LlamaModelName is not None and len(LlamaModelName) > 0 else "Meta-Llama-3-8B-Instruct-ct2-int8_float16/avan"
            selectedModel = next((modelConfig for modelConfig in self.app_config.models["Llama"] if modelConfig.name == selectedModelName), None)
            translationLang = get_lang_from_m2m100_name(LlamaLangName)

        if translationLang is not None:
            translationModel = TranslationModel(modelConfig=selectedModel, whisperLang=inputLang, translationLang=translationLang, batchSize=translationBatchSize, noRepeatNgramSize=translationNoRepeatNgramSize, numBeams=translationNumBeams, torchDtypeFloat16=translationTorchDtypeFloat16, usingBitsandbytes=translationUsingBitsandbytes)
           
        return translationLang, translationModel


def create_ui(app_config: ApplicationConfig):
    translateModelMd: str = None
    optionsMd: str = None
    readmeMd: str = None
    try:
        translateModelPath = pathlib.Path("docs/translateModel.md")
        with open(translateModelPath, "r", encoding="utf-8") as translateModelFile:
            translateModelMd = translateModelFile.read()
    except Exception as e:
        print("Error occurred during read translateModel.md file: ", str(e))
    try:
        optionsPath = pathlib.Path("docs/options.md")
        with open(optionsPath, "r", encoding="utf-8") as optionsFile:
            optionsMd = optionsFile.read()
    except Exception as e:
        print("Error occurred during read options.md file: ", str(e))
    try:
        with open("README.md", "r", encoding="utf-8") as readmeFile:
            readmeMd = readmeFile.read()
    except Exception as e:
        print("Error occurred during read options.md file: ", str(e))

    ui = WhisperTranscriber(app_config.input_audio_max_duration, app_config.vad_process_timeout, app_config.vad_cpu_cores, 
                            app_config.delete_uploaded_files, app_config.output_dir, app_config)

    # Specify a list of devices to use for parallel processing
    ui.set_parallel_devices(app_config.vad_parallel_devices)
    ui.set_auto_parallel(app_config.auto_parallel)

    is_whisper = False

    if app_config.whisper_implementation == "whisper":
        implementation_name = "Whisper"
        is_whisper = True
    elif app_config.whisper_implementation in ["faster-whisper", "faster_whisper"]:
        implementation_name = "Faster Whisper"
    else:
        # Try to convert from camel-case to title-case
        implementation_name = app_config.whisper_implementation.title().replace("_", " ").replace("-", " ")

    uiDescription = implementation_name + " is a general-purpose speech recognition model. It is trained on a large dataset of diverse " 
    uiDescription += " audio and is also a multi-task model that can perform multilingual speech recognition "
    uiDescription += " as well as speech translation and language identification. "

    uiDescription += "\n\n\n\nFor longer audio files (>10 minutes) not in English, it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option."

    # Recommend faster-whisper
    if is_whisper:
        uiDescription += "\n\n\n\nFor faster inference on GPU, try [faster-whisper](https://huggingface.co/spaces/aadnk/faster-whisper-webui)."

    if app_config.input_audio_max_duration > 0:
        uiDescription += "\n\n" + "Max audio file length: " + str(app_config.input_audio_max_duration) + " s"

    uiArticle = "Read the [documentation here](https://gitlab.com/aadnk/whisper-webui/-/blob/main/docs/options.md)."

    whisper_models = app_config.get_model_names("whisper")
    nllb_models = app_config.get_model_names("nllb")
    m2m100_models = app_config.get_model_names("m2m100")
    mt5_models = app_config.get_model_names("mt5")
    ALMA_models = app_config.get_model_names("ALMA")
    madlad400_models = app_config.get_model_names("madlad400")
    seamless_models = app_config.get_model_names("seamless")
    Llama_models = app_config.get_model_names("Llama")
    if not torch.cuda.is_available(): # Loading only quantized or models with medium-low parameters in an environment without GPU support.
        nllb_models = list(filter(lambda nllb: any(name in nllb for name in ["-600M", "-1.3B", "-3.3B-ct2"]), nllb_models))
        m2m100_models = list(filter(lambda m2m100: "12B" not in m2m100, m2m100_models))
        ALMA_models = list(filter(lambda alma: "GGUF" in alma or "ct2" in alma, ALMA_models))
        madlad400_models = list(filter(lambda madlad400: "ct2" in madlad400, madlad400_models))

    common_whisper_inputs = lambda : {
        gr.Dropdown(label="Whisper - Model (for audio)", choices=whisper_models, value=app_config.default_model_name if app_config.default_model_name != None else (lambda : None), elem_id="whisperModelName"),
        gr.Dropdown(label="Whisper - Language", choices=sorted(get_lang_whisper_names()), value=app_config.language if app_config.language != None else (lambda : None), elem_id="whisperLangName"),
    }
    common_m2m100_inputs = lambda : {
        gr.Dropdown(label="M2M100 - Model (for translate)", choices=m2m100_models, value=lambda : None, elem_id="m2m100ModelName"),
        gr.Dropdown(label="M2M100 - Language", choices=sorted(get_lang_m2m100_names()), value=lambda : None, elem_id="m2m100LangName"),
    }
    common_nllb_inputs = lambda : {
        gr.Dropdown(label="NLLB - Model (for translate)", choices=nllb_models, value=lambda : None, elem_id="nllbModelName"),
        gr.Dropdown(label="NLLB - Language", choices=sorted(get_lang_nllb_names()), value=lambda : None, elem_id="nllbLangName"),
    }
    common_mt5_inputs = lambda : {
        gr.Dropdown(label="MT5 - Model (for translate)", choices=mt5_models, value=lambda : None, elem_id="mt5ModelName"),
        gr.Dropdown(label="MT5 - Language", choices=sorted(get_lang_m2m100_names(["en", "ja", "zh"])), value=lambda : None, elem_id="mt5LangName"),
    }
    common_ALMA_inputs = lambda : {
        gr.Dropdown(label="ALMA - Model (for translate)", choices=ALMA_models, value=lambda : None, elem_id="ALMAModelName"),
        gr.Dropdown(label="ALMA - Language", choices=sort_lang_by_whisper_codes(["en", "de", "cs", "is", "ru", "zh", "ja"]), value=lambda : None, elem_id="ALMALangName"),
    }
    common_madlad400_inputs = lambda : {
        gr.Dropdown(label="madlad400 - Model (for translate)", choices=madlad400_models, value=lambda : None, elem_id="madlad400ModelName"),
        gr.Dropdown(label="madlad400 - Language", choices=sorted(get_lang_m2m100_names()), value=lambda : None, elem_id="madlad400LangName"),
    }
    common_seamless_inputs = lambda : {
        gr.Dropdown(label="seamless - Model (for translate)", choices=seamless_models, value=lambda : None, elem_id="seamlessModelName"),
        gr.Dropdown(label="seamless - Language", choices=sorted(get_lang_seamlessT_Tx_names()), value=lambda : None, elem_id="seamlessLangName"),
    }
    common_Llama_inputs = lambda : {
        gr.Dropdown(label="Llama - Model (for translate)", choices=Llama_models, value=lambda : None, elem_id="LlamaModelName"),
        gr.Dropdown(label="Llama - Language", choices=sorted(get_lang_m2m100_names()), value=lambda : None, elem_id="LlamaLangName"),
    }
    
    common_translation_inputs = lambda : {
        gr.Number(label="Translation - Batch Size", precision=0, value=app_config.translation_batch_size, elem_id="translationBatchSize"),
        gr.Number(label="Translation - No Repeat Ngram Size", precision=0, value=app_config.translation_no_repeat_ngram_size, elem_id="translationNoRepeatNgramSize", info="Prevent repetitions of ngrams with this size (set 0 to disable)."),
        gr.Number(label="Translation - Num Beams", precision=0, value=app_config.translation_num_beams, elem_id="translationNumBeams", info="Beam size (1 for greedy search)."),
        gr.Checkbox(label="Translation - Torch Dtype float16", visible=torch.cuda.is_available(), value=app_config.translation_torch_dtype_float16, info="Load the float32 translation model with float16 when the system supports GPU (reducing VRAM usage, not applicable to models that have already been quantized, such as Ctranslate2, GPTQ, GGUF)", elem_id="translationTorchDtypeFloat16"),
        gr.Radio(label="Translation - Using Bitsandbytes", visible=torch.cuda.is_available(), choices=[None, "int8", "int4"], value=app_config.translation_using_bitsandbytes, info="Load the float32 translation model into mixed-8bit or 4bit precision quantized model when the system supports GPU (reducing VRAM usage, not applicable to models that have already been quantized, such as Ctranslate2, GPTQ, GGUF)", elem_id="translationUsingBitsandbytes"),
    }

    common_vad_inputs = lambda : {
        gr.Dropdown(choices=["none", "silero-vad", "silero-vad-skip-gaps", "silero-vad-expand-into-gaps", "periodic-vad"], value=app_config.default_vad, label="VAD", elem_id="vad"),
        gr.Number(label="VAD - Merge Window (s)", precision=0, value=app_config.vad_merge_window, elem_id="vadMergeWindow", info="If set, any adjacent speech sections that are at most this number of seconds apart will be automatically merged."),
        gr.Number(label="VAD - Max Merge Size (s)", precision=0, value=app_config.vad_max_merge_size, elem_id="vadMaxMergeSize", info="Disables merging of adjacent speech sections if they are this number of seconds long."),
        gr.Number(label="VAD - Process Timeout (s)", precision=0, value=app_config.vad_process_timeout, elem_id="vadPocessTimeout", info="This configures the number of seconds until a process is killed due to inactivity, freeing RAM and video memory. The default value is 30 minutes."),
    }
    
    common_word_timestamps_inputs = lambda : {
        gr.Checkbox(label="Word Timestamps", value=app_config.word_timestamps, elem_id="word_timestamps", info="Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment."),
        gr.Checkbox(label="Word Timestamps - Highlight Words", value=app_config.highlight_words, elem_id="highlight_words", info="if word_timestamps is True, underline each word as it is spoken in srt and vtt"),
    }
    
    common_segments_filter_inputs = lambda : {
        gr.Checkbox(label="Whisper Segments Filter", value=app_config.whisper_segments_filter, elem_id="whisperSegmentsFilter", info="Filter the results of Whisper transcribe with the following conditions. It is recommended to enable this feature when using the large-v3 model to avoid hallucinations.") if idx == 0 else
        gr.Text(label=f"Filter {idx}", value=filterStr, elem_id=f"whisperSegmentsFilter{idx}") for idx, filterStr in enumerate([""] + app_config.whisper_segments_filters)
    }

    has_diarization_libs = Diarization.has_libraries()

    if not has_diarization_libs:
        print("Diarization libraries not found - disabling diarization")
        app_config.diarization = False

    common_diarization_inputs = lambda : {
        gr.Checkbox(label="Diarization", value=app_config.diarization, interactive=has_diarization_libs, elem_id="diarization", info="Whether to perform speaker diarization"),
        gr.Number(label="Diarization - Speakers", precision=0, value=app_config.diarization_speakers, interactive=has_diarization_libs, elem_id="diarization_speakers", info="The number of speakers to detect"),
        gr.Number(label="Diarization - Min Speakers", precision=0, value=app_config.diarization_min_speakers, interactive=has_diarization_libs, elem_id="diarization_min_speakers", info="The minimum number of speakers to detect"),
        gr.Number(label="Diarization - Max Speakers", precision=0, value=app_config.diarization_max_speakers, interactive=has_diarization_libs, elem_id="diarization_max_speakers", info="The maximum number of speakers to detect")
    }
    
    common_output = lambda : [
        gr.File(label="Download", height=200, elem_id="outputDownload"),
        gr.Text(label="Transcription", autoscroll=False, show_copy_button=True, interactive=True, elem_id="outputTranscription", elem_classes="scroll-show"),
        gr.Text(label="Segments", autoscroll=False, show_copy_button=True, interactive=True, elem_id="outputSegments", elem_classes="scroll-show"),
        gr.Text(label="Filtered segment items", autoscroll=False, visible=False, show_copy_button=True, interactive=True, elem_id="outputFiltered", elem_classes="scroll-show"),
    ]
    
    css = """
.options {
    max-height: 65vh !important;
    overflow-y: auto !important;
}
.scroll-show textarea {
    overflow-y: auto !important;
    scrollbar-width: auto !important;
}
.scroll-show textarea::-webkit-scrollbar {
    all: initial !important;
    background: #f1f1f1 !important;
}
.scroll-show textarea::-webkit-scrollbar-thumb {
    all: initial !important;
    background: #a8a8a8 !important;
}
"""

    is_queue_mode = app_config.queue_concurrency_count is not None and app_config.queue_concurrency_count > 0

    def create_transcribe(uiDescription: str, isQueueMode: bool, isFull: bool = False):
        with gr.Blocks() as transcribe:
            translateInput = gr.State(value="m2m100") # [Gradio 5.x] TypeError: State.__init__() got an unexpected keyword argument 'elem_id'
            sourceInput = gr.State(value="urlData")
            translateInput.elem_id = "translateInput"
            sourceInput.elem_id = "sourceInput"
            gr.Markdown(uiDescription)
            with gr.Row():
                with gr.Column():
                    submitBtn = gr.Button("Submit", variant="primary")
                    with gr.Column():
                        with gr.Row():
                            inputDict = common_whisper_inputs()
                        with gr.Tab(label="M2M100") as m2m100Tab:
                            with gr.Row():
                                inputDict.update(common_m2m100_inputs())
                        with gr.Tab(label="NLLB") as nllbTab:
                            with gr.Row():
                                inputDict.update(common_nllb_inputs())
                        with gr.Tab(label="MT5") as mt5Tab:
                            with gr.Row():
                                inputDict.update(common_mt5_inputs())
                        with gr.Tab(label="ALMA") as almaTab:
                            with gr.Row():
                                inputDict.update(common_ALMA_inputs())
                        with gr.Tab(label="madlad400") as madlad400Tab:
                            with gr.Row():
                                inputDict.update(common_madlad400_inputs())
                        with gr.Tab(label="seamless") as seamlessTab:
                            with gr.Row():
                                inputDict.update(common_seamless_inputs())
                        with gr.Tab(label="Llama") as llamaTab:
                            with gr.Row():
                                inputDict.update(common_Llama_inputs())
                        m2m100Tab.select(fn=lambda: "m2m100", inputs = [], outputs= [translateInput] )
                        nllbTab.select(fn=lambda: "nllb", inputs = [], outputs= [translateInput] )
                        mt5Tab.select(fn=lambda: "mt5", inputs = [], outputs= [translateInput] )
                        almaTab.select(fn=lambda: "ALMA", inputs = [], outputs= [translateInput] )
                        madlad400Tab.select(fn=lambda: "madlad400", inputs = [], outputs= [translateInput] )
                        seamlessTab.select(fn=lambda: "seamless", inputs = [], outputs= [translateInput] )
                        llamaTab.select(fn=lambda: "Llama", inputs = [], outputs= [translateInput] )
                    with gr.Column():
                        with gr.Tab(label="URL") as UrlTab:
                            inputDict.update({gr.Text(label="URL (YouTube, etc.)", elem_id = "urlData")})
                        with gr.Tab(label="Upload") as UploadTab:
                            inputDict.update({gr.File(label="Upload Files", file_count="multiple", elem_id = "multipleFiles")})
                        with gr.Tab(label="Microphone") as MicTab: # [Gradio 5.x] TypeError: Audio.__init__() got an unexpected keyword argument 'source'
                            inputDict.update({gr.Audio(sources=["microphone"], type="filepath", label="Microphone Input", elem_id = "microphoneData")})
                        UrlTab.select(fn=lambda: "urlData", inputs = [], outputs= [sourceInput] )
                        UploadTab.select(fn=lambda: "multipleFiles", inputs = [], outputs= [sourceInput] )
                        MicTab.select(fn=lambda: "microphoneData", inputs = [], outputs= [sourceInput] )
                        inputDict.update({gr.Dropdown(choices=["transcribe", "translate"], label="Task", value=app_config.task, elem_id = "task", info="Select the task - either \"transcribe\" to transcribe the audio to text, or \"translate\" to translate it to English.")})
                        with gr.Accordion("VAD options", open=False):
                            inputDict.update(common_vad_inputs())
                            if isFull:
                                inputDict.update({
                                    gr.Number(label="VAD - Padding (s)", precision=None, value=app_config.vad_padding, elem_id = "vadPadding", info="The number of seconds (floating point) to add to the beginning and end of each speech section. Setting this to a number larger than zero ensures that Whisper is more likely to correctly transcribe a sentence in the beginning of a speech section. However, this also increases the probability of Whisper assigning the wrong timestamp to each transcribed line. The default value is 1 second."),
                                    gr.Number(label="VAD - Prompt Window (s)", precision=None, value=app_config.vad_prompt_window, elem_id = "vadPromptWindow", info="The text of a detected line will be included as a prompt to the next speech section, if the speech section starts at most this number of seconds after the line has finished. For instance, if a line ends at 10:00, and the next speech section starts at 10:04, the line's text will be included if the prompt window is 4 seconds or more (10:04 - 10:00 = 4 seconds)."),
                                    gr.Dropdown(choices=VAD_INITIAL_PROMPT_MODE_VALUES, label="VAD - Initial Prompt Mode", value=app_config.vad_initial_prompt_mode, elem_id = "vadInitialPromptMode", info="prepend_all_segments: prepend the initial prompt to each VAD segment, prepend_first_segment: just the first segment")})
                        with gr.Accordion("Word Timestamps options", open=False):
                            inputDict.update(common_word_timestamps_inputs())
                            if isFull:
                                inputDict.update({
                                    gr.Text(label="Word Timestamps - Prepend Punctuations", value=app_config.prepend_punctuations, elem_id = "prepend_punctuations", info="if word_timestamps is True, merge these punctuation symbols with the next word"),
                                    gr.Text(label="Word Timestamps - Append Punctuations", value=app_config.append_punctuations, elem_id = "append_punctuations", info="if word_timestamps is True, merge these punctuation symbols with the previous word")})
                        if isFull:
                            with gr.Accordion("Whisper Advanced options", open=False):
                                inputDict.update({
                                    gr.TextArea(label="Initial Prompt", elem_id = "initial_prompt", info="Optional text to provide as a prompt for the first window"),
                                    gr.Number(label="Temperature", value=app_config.temperature, elem_id = "temperature", info="Temperature to use for sampling"),
                                    gr.Number(label="Best Of - Non-zero temperature", value=app_config.best_of, precision=0, elem_id = "best_of", info="Number of candidates when sampling with non-zero temperature"),
                                    gr.Number(label="Beam Size - Zero temperature", value=app_config.beam_size, precision=0, elem_id = "beam_size", info="Number of beams in beam search, only applicable when temperature is zero"),
                                    gr.Number(label="Patience - Zero temperature", value=app_config.patience, elem_id = "patience", info="Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search"),
                                    gr.Number(label="Length Penalty - Any temperature", value=lambda : None if app_config.length_penalty is None else app_config.length_penalty, elem_id = "length_penalty", info="Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default"),
                                    gr.Text(label="Suppress Tokens - Comma-separated list of token IDs", value=app_config.suppress_tokens, elem_id = "suppress_tokens", info="Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations"),
                                    gr.Checkbox(label="Condition on previous text", value=app_config.condition_on_previous_text, elem_id = "condition_on_previous_text", info="If True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop"),
                                    gr.Checkbox(label="FP16", value=app_config.fp16, elem_id = "fp16", info="Whether to perform inference in fp16; True by default; It will be ignored in faster-whisper because it is already a quantized model."),
                                    gr.Number(label="Temperature increment on fallback", value=app_config.temperature_increment_on_fallback, elem_id = "temperature_increment_on_fallback", info="Temperature to increase when falling back when the decoding fails to meet either of the thresholds below"),
                                    gr.Number(label="Compression ratio threshold", value=app_config.compression_ratio_threshold, elem_id = "compression_ratio_threshold", info="If the gzip compression ratio is higher than this value, treat the decoding as failed"),
                                    gr.Number(label="Logprob threshold", value=app_config.logprob_threshold, elem_id = "logprob_threshold", info="If the average log probability is lower than this value, treat the decoding as failed"),
                                    gr.Number(label="No speech threshold", value=app_config.no_speech_threshold, elem_id = "no_speech_threshold", info="If the probability of the <no-speech> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence"),
                                    })
                                if app_config.whisper_implementation == "faster-whisper":
                                    inputDict.update({
                                        gr.Number(label="Repetition Penalty", value=app_config.repetition_penalty, elem_id = "repetition_penalty", info="[faster-whisper] The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0."),
                                        gr.Number(label="No Repeat Ngram Size", value=app_config.no_repeat_ngram_size, precision=0, elem_id = "no_repeat_ngram_size", info="[faster-whisper] The model ensures that a sequence of words of no_repeat_ngram_size isn’t repeated in the output sequence. If specified, it must be a positive integer greater than 1.")
                                    })
                        with gr.Accordion("Whisper Segments Filter options", open=False):
                            inputDict.update(common_segments_filter_inputs())
                        with gr.Accordion("Diarization options", open=False):
                            inputDict.update(common_diarization_inputs())
                        with gr.Accordion("Translation options", open=False):
                            inputDict.update(common_translation_inputs())
                with gr.Column():
                    outputs = common_output()
            gr.Markdown(uiArticle)
            if optionsMd is not None:
                with gr.Accordion("docs/options.md", open=False):    
                    gr.Markdown(optionsMd)
            if translateModelMd is not None:
                with gr.Accordion("docs/translateModel.md", open=False):    
                    gr.Markdown(translateModelMd)
            if readmeMd is not None:
                with gr.Accordion("README.md", open=False):    
                    gr.Markdown(readmeMd)
                    
            inputDict.update({translateInput, sourceInput})
            submitBtn.click(fn=ui.transcribe_entry_progress if isQueueMode else ui.transcribe_entry,
                        inputs=inputDict, outputs=outputs)
            
        return transcribe

    def find_free_port() -> int:
        server_name=app_config.server_name
        server_port=app_config.server_port

        if server_name is None:
            server_name = '127.0.0.1'

        if server_port is None:
            server_port = 7860

        import socket
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((server_name, server_port))
                    return server_port
                except OSError:
                    server_port += 1

    def create_translation(isQueueMode: bool):
        with gr.Blocks() as translation:
            translateInput = gr.State(value="m2m100") # [Gradio 5.x] TypeError: State.__init__() got an unexpected keyword argument 'elem_id'
            translateInput.elem_id = "translateInput"
            with gr.Row():
                with gr.Column():
                    submitBtn = gr.Button("Submit", variant="primary")
                    with gr.Column():
                        with gr.Tab(label="M2M100") as m2m100Tab:
                            with gr.Row():
                                inputDict = common_m2m100_inputs()
                        with gr.Tab(label="NLLB") as nllbTab:
                            with gr.Row():
                                inputDict.update(common_nllb_inputs())
                        with gr.Tab(label="MT5") as mt5Tab:
                            with gr.Row():
                                inputDict.update(common_mt5_inputs())
                        with gr.Tab(label="ALMA") as almaTab:
                            with gr.Row():
                                inputDict.update(common_ALMA_inputs())
                        with gr.Tab(label="madlad400") as madlad400Tab:
                            with gr.Row():
                                inputDict.update(common_madlad400_inputs())
                        with gr.Tab(label="seamless") as seamlessTab:
                            with gr.Row():
                                inputDict.update(common_seamless_inputs())
                        with gr.Tab(label="Llama") as llamaTab:
                            with gr.Row():
                                inputDict.update(common_Llama_inputs())
                        m2m100Tab.select(fn=lambda: "m2m100", inputs = [], outputs= [translateInput] )
                        nllbTab.select(fn=lambda: "nllb", inputs = [], outputs= [translateInput] )
                        mt5Tab.select(fn=lambda: "mt5", inputs = [], outputs= [translateInput] )
                        almaTab.select(fn=lambda: "ALMA", inputs = [], outputs= [translateInput] )
                        madlad400Tab.select(fn=lambda: "madlad400", inputs = [], outputs= [translateInput] )
                        seamlessTab.select(fn=lambda: "seamless", inputs = [], outputs= [translateInput] )
                        llamaTab.select(fn=lambda: "Llama", inputs = [], outputs= [translateInput] )
                    with gr.Column():
                        inputDict.update({
                            gr.Dropdown(label="Input - Language", choices=sorted(get_lang_whisper_names()), value=app_config.language if app_config.language != None else (lambda : None), elem_id="inputLangName"),
                            gr.Text(lines=5, label="Input - Text", elem_id="inputText", elem_classes="scroll-show"),
                        })
                    with gr.Column():
                        with gr.Accordion("Translation options", open=False):
                            inputDict.update(common_translation_inputs())
                            inputDict.update({ gr.Checkbox(label="Translation - Enbale bilingual", value=True, info="Determines whether to enable bilingual translation results", elem_id="translationEnbaleBilingual"),
                                               gr.Checkbox(label="Translation - Detect line breaks", value=False, info="Determines whether to enable detecting line breaks in the text. If enabled, it will concatenate lines before translation", elem_id="translationDetectLineBreaks"),})
                with gr.Column():
                    outputs = [gr.Text(label="Translation Text", autoscroll=False, show_copy_button=True, interactive=True, elem_id="outputTranslationText", elem_classes="scroll-show"),]
            if translateModelMd is not None:
                with gr.Accordion("docs/translateModel.md", open=False):    
                    gr.Markdown(translateModelMd)
                    
            inputDict.update({translateInput})
            submitBtn.click(fn=ui.translation_entry_progress if isQueueMode else ui.translation_entry,
                        inputs=inputDict, outputs=outputs)
            
        return translation

    simpleTranscribe = create_transcribe(uiDescription, is_queue_mode)
    fullDescription = uiDescription + "\n\n\n\n" + "Be careful when changing some of the options in the full interface - this can cause the model to crash."
    fullTranscribe = create_transcribe(fullDescription, is_queue_mode, True)
    uiTranslation = create_translation(is_queue_mode)

    demo = gr.TabbedInterface([simpleTranscribe, fullTranscribe, uiTranslation], tab_names=["Simple", "Full", "Translation"], css=css)

    # Queue up the demo
    if is_queue_mode: # [Gradio 5.x] TypeError: Blocks.queue() got an unexpected keyword argument 'concurrency_count'
        demo.queue(default_concurrency_limit=app_config.queue_concurrency_count)
        print("Queue mode enabled (concurrency count: " + str(app_config.queue_concurrency_count) + ")")
    else:
        print("Queue mode disabled - progress bars will not be shown.")

    demo.launch(inbrowser=app_config.autolaunch, share=app_config.share, server_name=app_config.server_name, server_port=find_free_port())
               # ,ssr_mode=False) # [Gradio 5.x] ERROR: Exception in ASGI application
    
    # Clean up
    ui.close()

if __name__ == '__main__':
    default_app_config = ApplicationConfig.create_default()
    whisper_models = default_app_config.get_model_names("whisper")

    # Environment variable overrides
    default_whisper_implementation = os.environ.get("WHISPER_IMPLEMENTATION", default_app_config.whisper_implementation)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_audio_max_duration", type=int, default=default_app_config.input_audio_max_duration, \
                        help="Maximum audio file length in seconds, or -1 for no limit.") # 600
    parser.add_argument("--share", type=bool, default=default_app_config.share, \
                        help="True to share the app on HuggingFace.") # False
    parser.add_argument("--server_name", type=str, default=default_app_config.server_name, \
                        help="The host or IP to bind to. If None, bind to localhost.") # None
    parser.add_argument("--server_port", type=int, default=default_app_config.server_port, \
                        help="The port to bind to.") # 7860
    parser.add_argument("--queue_concurrency_count", type=int, default=default_app_config.queue_concurrency_count, \
                        help="The number of concurrent requests to process.") # 1
    parser.add_argument("--default_model_name", type=str, choices=whisper_models, default=default_app_config.default_model_name, \
                        help="The default model name.") # medium
    parser.add_argument("--default_vad", type=str, default=default_app_config.default_vad, \
                        help="The default VAD.") # silero-vad
    parser.add_argument("--vad_initial_prompt_mode", type=str, default=default_app_config.vad_initial_prompt_mode, choices=VAD_INITIAL_PROMPT_MODE_VALUES, \
                        help="Whether or not to prepend the initial prompt to each VAD segment (prepend_all_segments), or just the first segment (prepend_first_segment)") # prepend_first_segment
    parser.add_argument("--vad_parallel_devices", type=str, default=default_app_config.vad_parallel_devices, \
                        help="A commma delimited list of CUDA devices to use for parallel processing. If None, disable parallel processing.") # ""
    parser.add_argument("--vad_cpu_cores", type=int, default=default_app_config.vad_cpu_cores, \
                        help="The number of CPU cores to use for VAD pre-processing.") # 1
    parser.add_argument("--vad_process_timeout", type=float, default=default_app_config.vad_process_timeout, \
                        help="The number of seconds before inactivate processes are terminated. Use 0 to close processes immediately, or None for no timeout.") # 1800
    parser.add_argument("--auto_parallel", type=bool, default=default_app_config.auto_parallel, \
                        help="True to use all available GPUs and CPU cores for processing. Use vad_cpu_cores/vad_parallel_devices to specify the number of CPU cores/GPUs to use.") # False
    parser.add_argument("--output_dir", "-o", type=str, default=default_app_config.output_dir, \
                        help="directory to save the outputs")
    parser.add_argument("--whisper_implementation", type=str, default=default_whisper_implementation, choices=["whisper", "faster-whisper"],\
                        help="the Whisper implementation to use")
    parser.add_argument("--compute_type", type=str, default=default_app_config.compute_type, choices=["default", "auto", "int8", "int8_float16", "int16", "float16", "float32"], \
                        help="the compute type to use for inference")
    parser.add_argument("--threads", type=optional_int, default=0, 
                        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    
    parser.add_argument("--vad_max_merge_size", type=int, default=default_app_config.vad_max_merge_size, \
                        help="The number of VAD - Max Merge Size (s).") # 30
    parser.add_argument("--language", type=str, default=None, choices=sorted(get_lang_whisper_names()) + sorted([k.title() for k in _TO_LANG_CODE_WHISPER.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--save_downloaded_files", action='store_true', \
                        help="True to move downloaded files to outputs directory. This argument will take effect only after output_dir is set.")
    parser.add_argument("--merge_subtitle_with_sources", action='store_true', \
                        help="True to merge subtitle(srt) with sources and move the sources files to the outputs directory. This argument will take effect only after output_dir is set.")
    parser.add_argument("--input_max_file_name_length", type=int, default=100, \
                        help="Maximum length of a file name.")
    parser.add_argument("--autolaunch", action='store_true', \
                        help="open the webui URL in the system's default browser upon launch")
    
    parser.add_argument('--auth_token', type=str, default=default_app_config.auth_token, help='HuggingFace API Token (optional)')
    parser.add_argument("--diarization", type=str2bool, default=default_app_config.diarization, \
                        help="whether to perform speaker diarization")
    parser.add_argument("--diarization_num_speakers", type=int, default=default_app_config.diarization_speakers, help="Number of speakers")
    parser.add_argument("--diarization_min_speakers", type=int, default=default_app_config.diarization_min_speakers, help="Minimum number of speakers")
    parser.add_argument("--diarization_max_speakers", type=int, default=default_app_config.diarization_max_speakers, help="Maximum number of speakers")
    parser.add_argument("--diarization_process_timeout", type=int, default=default_app_config.diarization_process_timeout, \
                        help="Number of seconds before inactivate diarization processes are terminated. Use 0 to close processes immediately, or None for no timeout.")

    args = parser.parse_args().__dict__

    updated_config = default_app_config.update(**args)

    updated_config.whisper_implementation = "faster-whisper"
    updated_config.input_audio_max_duration = -1
    updated_config.default_model_name = "large-v2"
    updated_config.output_dir = "output"
    updated_config.vad_max_merge_size = 90
    updated_config.merge_subtitle_with_sources = False
    updated_config.autolaunch = True
    updated_config.auto_parallel = False
    updated_config.save_downloaded_files = True
    
    try:
        if torch.cuda.is_available():
            deviceId = torch.cuda.current_device()
            totalVram = torch.cuda.get_device_properties(deviceId).total_memory
            print(f"Total Vram: {totalVram/(1024*1024*1024):.4f}G")
            if totalVram/(1024*1024*1024) <= 4: #VRAM <= 4 GB
                updated_config.vad_process_timeout = 0
    except Exception as e:
        print(traceback.format_exc())
        print("Error detect vram: " + str(e))

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    print("Using whisper implementation: " + updated_config.whisper_implementation)
    create_ui(app_config=updated_config)