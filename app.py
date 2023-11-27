from datetime import datetime
import json
import math
from typing import Iterator, Union
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
from src.languages import _TO_LANGUAGE_CODE, get_language_names, get_language_from_name, get_language_from_code
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
from src.utils import optional_int, slugify, str2bool, write_srt, write_vtt
from src.vad import AbstractTranscription, NonSpeechStrategy, PeriodicTranscriptionConfig, TranscriptionConfig, VadPeriodicTranscription, VadSileroTranscription
from src.whisper.abstractWhisperContainer import AbstractWhisperContainer
from src.whisper.whisperFactory import create_whisper_container
from src.nllb.nllbModel import NllbModel
from src.nllb.nllbLangs import _TO_NLLB_LANG_CODE
from src.nllb.nllbLangs import get_nllb_lang_names
from src.nllb.nllbLangs import get_nllb_lang_from_name

import shutil
import zhconv
import tqdm

# Configure more application defaults in config.json5

# Gradio seems to truncate files without keeping the extension, so we need to truncate the file prefix ourself 
MAX_FILE_PREFIX_LENGTH = 17

# Limit auto_parallel to a certain number of CPUs (specify vad_cpu_cores to get a higher number)
MAX_AUTO_CPU_CORES = 8

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]

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

    # Entry function for the simple tab
    def transcribe_webui_simple(self, modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, 
                                vad, vadMergeWindow, vadMaxMergeSize, 
                                word_timestamps: bool = False, highlight_words: bool = False,
                                diarization: bool = False, diarization_speakers: int = 2,
                                diarization_min_speakers = 1, diarization_max_speakers = 8):
        return self.transcribe_webui_simple_progress(modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, 
                                vad, vadMergeWindow, vadMaxMergeSize, 
                                word_timestamps, highlight_words,
                                diarization, diarization_speakers,
                                diarization_min_speakers, diarization_max_speakers)
    
    # Entry function for the simple tab progress
    def transcribe_webui_simple_progress(self, modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, 
                                         vad, vadMergeWindow, vadMaxMergeSize, 
                                         word_timestamps: bool = False, highlight_words: bool = False, 
                                         diarization: bool = False, diarization_speakers: int = 2,
                                        diarization_min_speakers = 1, diarization_max_speakers = 8,
                                         progress=gr.Progress()):
        
        vadOptions = VadOptions(vad, vadMergeWindow, vadMaxMergeSize, self.app_config.vad_padding, self.app_config.vad_prompt_window, self.app_config.vad_initial_prompt_mode)

        if diarization:
            if diarization_speakers is not None and diarization_speakers < 1:
                self.set_diarization(auth_token=self.app_config.auth_token, min_speakers=diarization_min_speakers, max_speakers=diarization_max_speakers)
            else:
                self.set_diarization(auth_token=self.app_config.auth_token, num_speakers=diarization_speakers, min_speakers=diarization_min_speakers, max_speakers=diarization_max_speakers)
        else:
            self.unset_diarization()

        return self.transcribe_webui(modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, vadOptions, 
                                     word_timestamps=word_timestamps, highlight_words=highlight_words, progress=progress)

    # Entry function for the full tab
    def transcribe_webui_full(self, modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, 
                              vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow, vadInitialPromptMode, 
                              # Word timestamps
                              word_timestamps: bool, highlight_words: bool, prepend_punctuations: str, append_punctuations: str,
                              initial_prompt: str, temperature: float, best_of: int, beam_size: int, patience: float, length_penalty: float, suppress_tokens: str, 
                              condition_on_previous_text: bool, fp16: bool, temperature_increment_on_fallback: float, 
                              compression_ratio_threshold: float, logprob_threshold: float, no_speech_threshold: float,
                              diarization: bool = False, diarization_speakers: int = 2,
                              diarization_min_speakers = 1, diarization_max_speakers = 8):
        
        return self.transcribe_webui_full_progress(modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, 
                                vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow, vadInitialPromptMode,
                                word_timestamps, highlight_words, prepend_punctuations, append_punctuations,
                                initial_prompt, temperature, best_of, beam_size, patience, length_penalty, suppress_tokens,
                                condition_on_previous_text, fp16, temperature_increment_on_fallback,
                                compression_ratio_threshold, logprob_threshold, no_speech_threshold,
                                diarization, diarization_speakers,
                                diarization_min_speakers, diarization_max_speakers)

    # Entry function for the full tab with progress
    def transcribe_webui_full_progress(self, modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, 
                                        vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow, vadInitialPromptMode,
                                        # Word timestamps
                                        word_timestamps: bool, highlight_words: bool, prepend_punctuations: str, append_punctuations: str,   
                                        initial_prompt: str, temperature: float, best_of: int, beam_size: int, patience: float, length_penalty: float, suppress_tokens: str, 
                                        condition_on_previous_text: bool, fp16: bool, temperature_increment_on_fallback: float, 
                                        compression_ratio_threshold: float, logprob_threshold: float, no_speech_threshold: float, 
                                        diarization: bool = False, diarization_speakers: int = 2, 
                                        diarization_min_speakers = 1, diarization_max_speakers = 8,
                                        progress=gr.Progress()):

        # Handle temperature_increment_on_fallback
        if temperature_increment_on_fallback is not None:
            temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
        else:
            temperature = [temperature]

        vadOptions = VadOptions(vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow, vadInitialPromptMode)

        # Set diarization
        if diarization:
            if diarization_speakers is not None and diarization_speakers < 1:
                self.set_diarization(auth_token=self.app_config.auth_token, min_speakers=diarization_min_speakers, max_speakers=diarization_max_speakers)
            else:
                self.set_diarization(auth_token=self.app_config.auth_token, num_speakers=diarization_speakers, min_speakers=diarization_min_speakers, max_speakers=diarization_max_speakers)
        else:
            self.unset_diarization()

        return self.transcribe_webui(modelName, languageName, nllbModelName, nllbLangName, urlData, multipleFiles, microphoneData, task, vadOptions,
                                     initial_prompt=initial_prompt, temperature=temperature, best_of=best_of, beam_size=beam_size, patience=patience, length_penalty=length_penalty, suppress_tokens=suppress_tokens,
                                     condition_on_previous_text=condition_on_previous_text, fp16=fp16,
                                     compression_ratio_threshold=compression_ratio_threshold, logprob_threshold=logprob_threshold, no_speech_threshold=no_speech_threshold, 
                                     word_timestamps=word_timestamps, prepend_punctuations=prepend_punctuations, append_punctuations=append_punctuations, highlight_words=highlight_words,
                                     progress=progress)

    def transcribe_webui(self, modelName: str, languageName: str, nllbModelName: str, nllbLangName: str, urlData: str, multipleFiles, microphoneData: str, task: str, 
                         vadOptions: VadOptions, progress: gr.Progress = None, highlight_words: bool = False, 
                         **decodeOptions: dict):
        try:
            progress(0, desc="init audio sources")
            sources = self.__get_source(urlData, multipleFiles, microphoneData)
            if (len(sources) == 0):
                raise Exception("init audio sources failed...")
            try:
                progress(0, desc="init whisper model")
                whisper_lang = get_language_from_name(languageName)
                selectedLanguage = languageName.lower() if languageName is not None and len(languageName) > 0 else None
                selectedModel = modelName if modelName is not None else "base"

                model = create_whisper_container(whisper_implementation=self.app_config.whisper_implementation, 
                                                 model_name=selectedModel, compute_type=self.app_config.compute_type, 
                                                 cache=self.model_cache, models=self.app_config.models)
                
                progress(0, desc="init translate model")
                nllb_lang = get_nllb_lang_from_name(nllbLangName)
                selectedNllbModelName = nllbModelName if nllbModelName is not None and len(nllbModelName) > 0 else "nllb-200-distilled-600M/facebook"
                selectedNllbModel = next((modelConfig for modelConfig in self.app_config.nllb_models if modelConfig.name == selectedNllbModelName), None)
                
                nllb_model = NllbModel(model_config=selectedNllbModel, whisper_lang=whisper_lang, nllb_lang=nllb_lang) # load_model=True
                
                progress(0, desc="init transcribe")
                # Result
                download = []
                zip_file_lookup = {}
                text = ""
                vtt = ""

                # Write result
                downloadDirectory = tempfile.mkdtemp()
                source_index = 0
                extra_tasks_count = 1 if nllb_lang is not None else 0

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
                    result = self.transcribe_file(model, source.source_path, selectedLanguage, task, vadOptions, scaled_progress_listener, **decodeOptions)
                    if whisper_lang is None and result["language"] is not None and len(result["language"]) > 0:
                        whisper_lang = get_language_from_code(result["language"])
                        nllb_model.whisper_lang = whisper_lang
                        
                    short_name, suffix = source.get_short_name_suffix(max_length=self.app_config.input_max_file_name_length)
                    filePrefix = slugify(source_prefix + short_name, allow_unicode=True)

                    # Update progress
                    current_progress += source_audio_duration

                    source_download, source_text, source_vtt = self.write_result(result, nllb_model, filePrefix + suffix.replace(".", "_"), outputDirectory, highlight_words, scaled_progress_listener)

                    if self.app_config.merge_subtitle_with_sources and self.app_config.output_dir is not None:
                        print("\nmerge subtitle(srt) with source file [" + source.source_name + "]\n")
                        outRsult = ""
                        try:
                            srt_path = source_download[0]
                            save_path = os.path.join(self.app_config.output_dir, filePrefix)
                            # save_without_ext, ext = os.path.splitext(save_path)
                            source_lang = "." + whisper_lang.code if whisper_lang is not None else ""
                            translate_lang = "." + nllb_lang.code if nllb_lang is not None else ""
                            output_with_srt = save_path + source_lang + translate_lang + suffix
        
                            #ffmpeg -i "input.mp4" -i "input.srt" -c copy -c:s mov_text output.mp4
                            input_file = ffmpeg.input(source.source_path)
                            input_srt = ffmpeg.input(srt_path)
                            out = ffmpeg.output(input_file, input_srt, output_with_srt, vcodec='copy', acodec='copy', scodec='mov_text')
                            outRsult = out.run(overwrite_output=True)
                        except Exception as e:
                            # Ignore error - it's just a cleanup
                            print("Error merge subtitle with source file: \n" + source.source_path + ", \n" + str(e), outRsult)
                    elif self.app_config.save_downloaded_files and self.app_config.output_dir is not None and urlData:
                        print("Saving downloaded file [" + source.source_name + "]")
                        try:
                            save_path = os.path.join(self.app_config.output_dir, filePrefix)
                            shutil.copy(source.source_path, save_path + suffix)
                        except Exception as e:
                            # Ignore error - it's just a cleanup
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

                return download, text, vtt

            finally:
                # Cleanup source
                if self.deleteUploadedFiles:
                    for source in sources:
                        print("Deleting temporary source file: " + source.source_path)
                        try:
                            os.remove(source.source_path)
                        except Exception as e:
                            # Ignore error - it's just a cleanup
                            print("Error deleting temporary source file: \n" + source.source_path + ", \n" + str(e))
        
        except ExceededMaximumDuration as e:
            return [], ("[ERROR]: Maximum remote video length is " + str(e.maxDuration) + "s, file was " + str(e.videoDuration) + "s"), "[ERROR]"
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return [], ("Error occurred during transcribe: " + str(e)), ""
        

    def transcribe_file(self, model: AbstractWhisperContainer, audio_path: str, language: str, task: str = None, 
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
        whisperCallable = model.create_callback(language, task, prompt_strategy=prompt_strategy, **decodeOptions)

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
            self.vad_model = VadSileroTranscription()

        config = TranscriptionConfig(non_speech_strategy = non_speech_strategy, 
                max_silent_period=vadOptions.vadMergeWindow, max_merge_size=vadOptions.vadMaxMergeSize, 
                segment_padding_left=vadOptions.vadPadding, segment_padding_right=vadOptions.vadPadding, 
                max_prompt_window=vadOptions.vadPromptWindow)

        return config

    def write_result(self, result: dict, nllb_model: NllbModel, source_name: str, output_dir: str, highlight_words: bool = False, progressListener: ProgressListener = None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        text = result["text"]
        segments = result["segments"]
        language = result["language"]
        languageMaxLineWidth = self.__get_max_line_width(language)

        if nllb_model.nllb_lang is not None:
            try:
                segments_progress_listener = SubTaskProgressListener(progressListener, 
                                               base_task_total=progressListener.sub_task_total, 
                                               sub_task_start=1, 
                                               sub_task_total=1)
                pbar = tqdm.tqdm(total=len(segments))
                perf_start_time = time.perf_counter()
                nllb_model.load_model()
                for idx, segment in enumerate(segments):
                    seg_text = segment["text"]
                    if language == "zh":
                        segment["text"] = zhconv.convert(seg_text, "zh-tw")
                    if nllb_model.nllb_lang is not None:
                        segment["text"] = nllb_model.translation(seg_text)
                    pbar.update(1)
                    segments_progress_listener.on_progress(idx+1, len(segments), desc=f"Process segments: {idx}/{len(segments)}")

                nllb_model.release_vram()
                perf_end_time = time.perf_counter()
                # Call the finished callback
                if segments_progress_listener is not None:
                    segments_progress_listener.on_finished(desc=f"Process segments: {idx}/{len(segments)}")

                print("\n\nprocess segments took {} seconds.\n\n".format(perf_end_time - perf_start_time))
            except Exception as e:
                # Ignore error - it's just a cleanup
                print("Error process segments: " + str(e))

        print("Max line width " + str(languageMaxLineWidth) + " for language:" + language)
        vtt = self.__get_subs(result["segments"], "vtt", languageMaxLineWidth, highlight_words=highlight_words)
        srt = self.__get_subs(result["segments"], "srt", languageMaxLineWidth, highlight_words=highlight_words)
        json_result = json.dumps(result, indent=4, ensure_ascii=False)

        if language == "zh" or (nllb_model.nllb_lang is not None and nllb_model.nllb_lang.code == "zho_Hant"):
            vtt = zhconv.convert(vtt, "zh-tw")
            srt = zhconv.convert(srt, "zh-tw")
            text = zhconv.convert(text, "zh-tw")
            json_result = zhconv.convert(json_result, "zh-tw")

        output_files = []
        output_files.append(self.__create_file(srt, output_dir, source_name + "-subs.srt"));
        output_files.append(self.__create_file(vtt, output_dir, source_name + "-subs.vtt"));
        output_files.append(self.__create_file(text, output_dir, source_name + "-transcript.txt"));
        output_files.append(self.__create_file(json_result, output_dir, source_name + "-result.json"));

        return output_files, text, vtt

    def clear_cache(self):
        self.model_cache.clear()
        self.vad_model = None

    def __get_source(self, urlData, multipleFiles, microphoneData):
        return get_audio_source_collection(urlData, multipleFiles, microphoneData, self.inputAudioMaxDuration)

    def __get_max_line_width(self, language: str) -> int:
        if (language and language.lower() in ["japanese", "ja", "chinese", "zh"]):
            # Chinese characters and kana are wider, so limit line length to 40 characters
            return 40
        else:
            # TODO: Add more languages
            # 80 latin characters should fit on a 1080p/720p screen
            return 80

    def __get_subs(self, segments: Iterator[dict], format: str, maxLineWidth: int, highlight_words: bool = False) -> str:
        segmentStream = StringIO()

        if format == 'vtt':
            write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth, highlight_words=highlight_words)
        elif format == 'srt':
            write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth, highlight_words=highlight_words)
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

def create_ui(app_config: ApplicationConfig):
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

    ui_description = implementation_name + " is a general-purpose speech recognition model. It is trained on a large dataset of diverse " 
    ui_description += " audio and is also a multi-task model that can perform multilingual speech recognition "
    ui_description += " as well as speech translation and language identification. "

    ui_description += "\n\n\n\nFor longer audio files (>10 minutes) not in English, it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option."

    # Recommend faster-whisper
    if is_whisper:
        ui_description += "\n\n\n\nFor faster inference on GPU, try [faster-whisper](https://huggingface.co/spaces/aadnk/faster-whisper-webui)."

    if app_config.input_audio_max_duration > 0:
        ui_description += "\n\n" + "Max audio file length: " + str(app_config.input_audio_max_duration) + " s"

    ui_article = "Read the [documentation here](https://gitlab.com/aadnk/whisper-webui/-/blob/main/docs/options.md)."
    ui_article += "\n\nWhisper's Task 'translate' only implements the functionality of translating other languages into English. "
    ui_article += "OpenAI does not guarantee translations between arbitrary languages. In such cases, you can choose to use the NLLB Model to implement the translation task. "
    ui_article += "However, it's important to note that the NLLB Model runs slowly, and the completion time may be twice as long as usual. "
    ui_article += "\n\nThe larger the parameters of the NLLB model, the better its performance is expected to be. "
    ui_article += "However, it also requires higher computational resources, making it slower to operate. "
    ui_article += "On the other hand, the version converted from ct2 (CTranslate2) requires lower resources and operates at a faster speed."
    ui_article += "\n\nCurrently, enabling word-level timestamps cannot be used in conjunction with NLLB Model translation "
    ui_article += "because Word Timestamps will split the source text, and after translation, it becomes a non-word-level string. "
    ui_article += "\n\nThe 'mt5-zh-ja-en-trimmed' model is finetuned from Google's 'mt5-base' model. "
    ui_article += "This model has a relatively good translation speed, but it only supports three languages: Chinese, Japanese, and English. "

    whisper_models = app_config.get_model_names()
    nllb_models = app_config.get_nllb_model_names()
    
    common_whisper_inputs = lambda : [
        gr.Dropdown(label="Whisper Model (for audio)", choices=whisper_models, value=app_config.default_model_name),
        gr.Dropdown(label="Whisper Language", choices=sorted(get_language_names()), value=app_config.language),
    ]
    common_nllb_inputs = lambda : [
        gr.Dropdown(label="NLLB Model (for translate)", choices=nllb_models),
        gr.Dropdown(label="NLLB Language", choices=sorted(get_nllb_lang_names())),
    ]
    common_audio_inputs = lambda : [
        gr.Text(label="URL (YouTube, etc.)"),
        gr.File(label="Upload Files", file_count="multiple"),
        gr.Audio(source="microphone", type="filepath", label="Microphone Input"),
        gr.Dropdown(choices=["transcribe", "translate"], label="Task", value=app_config.task),
    ]

    common_vad_inputs = lambda : [
        gr.Dropdown(choices=["none", "silero-vad", "silero-vad-skip-gaps", "silero-vad-expand-into-gaps", "periodic-vad"], value=app_config.default_vad, label="VAD"),
        gr.Number(label="VAD - Merge Window (s)", precision=0, value=app_config.vad_merge_window),
        gr.Number(label="VAD - Max Merge Size (s)", precision=0, value=app_config.vad_max_merge_size),
    ]
    
    common_word_timestamps_inputs = lambda : [
        gr.Checkbox(label="Word Timestamps", value=app_config.word_timestamps),
        gr.Checkbox(label="Word Timestamps - Highlight Words", value=app_config.highlight_words),
    ]

    has_diarization_libs = Diarization.has_libraries()

    if not has_diarization_libs:
        print("Diarization libraries not found - disabling diarization")
        app_config.diarization = False

    common_diarization_inputs = lambda : [
        gr.Checkbox(label="Diarization", value=app_config.diarization, interactive=has_diarization_libs),
        gr.Number(label="Diarization - Speakers", precision=0, value=app_config.diarization_speakers, interactive=has_diarization_libs),
        gr.Number(label="Diarization - Min Speakers", precision=0, value=app_config.diarization_min_speakers, interactive=has_diarization_libs),
        gr.Number(label="Diarization - Max Speakers", precision=0, value=app_config.diarization_max_speakers, interactive=has_diarization_libs)
    ]
    
    common_output = lambda : [
        gr.File(label="Download"),
        gr.Text(label="Transcription", autoscroll=False),
        gr.Text(label="Segments", autoscroll=False),
    ]

    is_queue_mode = app_config.queue_concurrency_count is not None and app_config.queue_concurrency_count > 0

    simple_callback = gr.CSVLogger()

    with gr.Blocks() as simple_transcribe:
        gr.Markdown(ui_description)
        with gr.Row():
            with gr.Column():
                simple_submit = gr.Button("Submit", variant="primary")
                with gr.Column():
                    with gr.Row():
                        simple_input = common_whisper_inputs()
                    with gr.Row():
                        simple_input += common_nllb_inputs()
                with gr.Column():
                    simple_input += common_audio_inputs() + common_vad_inputs() + common_word_timestamps_inputs() + common_diarization_inputs()
            with gr.Column():
                simple_output = common_output()
                simple_flag = gr.Button("Flag")
        gr.Markdown(ui_article)

        # This needs to be called at some point prior to the first call to callback.flag()
        simple_callback.setup(simple_input + simple_output, "flagged")

        simple_submit.click(fn=ui.transcribe_webui_simple_progress if is_queue_mode else ui.transcribe_webui_simple,
                    inputs=simple_input, outputs=simple_output)
        # We can choose which components to flag -- in this case, we'll flag all of them
        simple_flag.click(lambda *args: print("simple_callback.flag...") or simple_callback.flag(args), simple_input + simple_output, None, preprocess=False)

    full_description = ui_description + "\n\n\n\n" + "Be careful when changing some of the options in the full interface - this can cause the model to crash."

    full_callback = gr.CSVLogger()

    with gr.Blocks() as full_transcribe:
        gr.Markdown(full_description)
        with gr.Row():
            with gr.Column():
                full_submit = gr.Button("Submit", variant="primary")
                with gr.Column():
                    with gr.Row():
                        full_input1 = common_whisper_inputs()
                    with gr.Row():
                        full_input1 += common_nllb_inputs()
                with gr.Column():
                    full_input1 += common_audio_inputs() + common_vad_inputs() + [
                    gr.Number(label="VAD - Padding (s)", precision=None, value=app_config.vad_padding),
                    gr.Number(label="VAD - Prompt Window (s)", precision=None, value=app_config.vad_prompt_window),
                    gr.Dropdown(choices=VAD_INITIAL_PROMPT_MODE_VALUES, label="VAD - Initial Prompt Mode")]

                    full_input2 = common_word_timestamps_inputs() + [
                    gr.Text(label="Word Timestamps - Prepend Punctuations", value=app_config.prepend_punctuations),
                    gr.Text(label="Word Timestamps - Append Punctuations", value=app_config.append_punctuations),
                    gr.TextArea(label="Initial Prompt"),
                    gr.Number(label="Temperature", value=app_config.temperature),
                    gr.Number(label="Best Of - Non-zero temperature", value=app_config.best_of, precision=0),
                    gr.Number(label="Beam Size - Zero temperature", value=app_config.beam_size, precision=0),
                    gr.Number(label="Patience - Zero temperature", value=app_config.patience),
                    gr.Number(label="Length Penalty - Any temperature", value=app_config.length_penalty),
                    gr.Text(label="Suppress Tokens - Comma-separated list of token IDs", value=app_config.suppress_tokens),
                    gr.Checkbox(label="Condition on previous text", value=app_config.condition_on_previous_text),
                    gr.Checkbox(label="FP16", value=app_config.fp16),
                    gr.Number(label="Temperature increment on fallback", value=app_config.temperature_increment_on_fallback),
                    gr.Number(label="Compression ratio threshold", value=app_config.compression_ratio_threshold),
                    gr.Number(label="Logprob threshold", value=app_config.logprob_threshold),
                    gr.Number(label="No speech threshold", value=app_config.no_speech_threshold)] + common_diarization_inputs()

            with gr.Column():
                full_output = common_output()
                full_flag = gr.Button("Flag")
        gr.Markdown(ui_article)

        # This needs to be called at some point prior to the first call to callback.flag()
        full_callback.setup(full_input1 + full_input2 + full_output, "flagged")

        full_submit.click(fn=ui.transcribe_webui_full_progress if is_queue_mode else ui.transcribe_webui_full,
                    inputs=full_input1+full_input2, outputs=full_output)
        # We can choose which components to flag -- in this case, we'll flag all of them
        full_flag.click(lambda *args: print("full_callback.flag...") or full_callback.flag(args), full_input1 + full_input2 + full_output, None, preprocess=False)

    demo = gr.TabbedInterface([simple_transcribe, full_transcribe], tab_names=["Simple", "Full"])

    # Queue up the demo
    if is_queue_mode:
        demo.queue(concurrency_count=app_config.queue_concurrency_count)
        print("Queue mode enabled (concurrency count: " + str(app_config.queue_concurrency_count) + ")")
    else:
        print("Queue mode disabled - progress bars will not be shown.")
   
    demo.launch(inbrowser=app_config.autolaunch, share=app_config.share, server_name=app_config.server_name, server_port=app_config.server_port)
    
    # Clean up
    ui.close()

if __name__ == '__main__':
    default_app_config = ApplicationConfig.create_default()
    whisper_models = default_app_config.get_model_names()
    nllb_models = default_app_config.get_nllb_model_names()

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
    parser.add_argument("--language", type=str, default=None, choices=sorted(get_language_names()) + sorted([k.title() for k in _TO_LANGUAGE_CODE.keys()]),
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

    # updated_config.whisper_implementation = "faster-whisper"
    # updated_config.input_audio_max_duration = -1
    # updated_config.default_model_name = "large-v2"
    # updated_config.output_dir = "output"
    # updated_config.vad_max_merge_size = 90
    # updated_config.merge_subtitle_with_sources = False
    # updated_config.autolaunch = True
    # updated_config.auto_parallel = False
    # updated_config.save_downloaded_files = True

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    print("Using whisper implementation: " + updated_config.whisper_implementation)
    create_ui(app_config=updated_config)