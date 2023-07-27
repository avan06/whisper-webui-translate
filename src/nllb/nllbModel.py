import os
import warnings
import huggingface_hub
import requests
import torch

import ctranslate2
import transformers

from typing import Optional
from src.config import ModelConfig
from src.languages import Language
from src.nllb.nllbLangs import NllbLang, get_nllb_lang_from_code_whisper

class NllbModel:
    def __init__(
        self,
        model_config: ModelConfig,
        device: str = None,
        whisper_lang: Language = None,
        nllb_lang: NllbLang = None,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        load_model: bool = False,
    ):
        """Initializes the Nllb-200 model.

        Args:
          model_config: Config of the model to use (distilled-600M, distilled-1.3B, 
            1.3B, 3.3B...) or a path to a converted
            model directory. When a size is configured, the converted model is downloaded
            from the Hugging Face Hub.
          device: Device to use for computation (cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, 
            ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia).
          device_index: Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of IDs
            (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
            when transcribe() is called from multiple Python threads (see also num_workers).
          compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
          cpu_threads: Number of threads to use when running on CPU (4 by default).
            A non zero value overrides the OMP_NUM_THREADS environment variable.
          num_workers: When transcribe() is called from multiple Python threads,
            having multiple workers enables true parallelism when running the model
            (concurrent calls to self.model.generate() will run in parallel).
            This can improve the global throughput at the cost of increased memory usage.
          download_root: Directory where the models should be saved. If not set, the models
            are saved in the standard Hugging Face cache directory.
          local_files_only:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
        """
        self.whisper_lang = whisper_lang
        self.nllb_whisper_lang = get_nllb_lang_from_code_whisper(whisper_lang.code.lower() if whisper_lang is not None else "en")
        self.nllb_lang = nllb_lang
        self.model_config = model_config

        if nllb_lang is None:
            return

        if os.path.isdir(model_config.url):
            self.model_path = model_config.url
        else:
            self.model_path = download_model(
                model_config,
                local_files_only=local_files_only,
                cache_dir=download_root,
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda" if "ct2" in self.model_path else "cuda:0"
            else:
                device = "cpu"

        self.device = device

        if load_model:
            self.load_model()

    def load_model(self):
        print('\n\nLoading model: %s\n\n' % self.model_path)
        if "ct2" in self.model_path:
            self.target_prefix = [self.nllb_lang.code]
            self.trans_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, src_lang=self.nllb_whisper_lang.code)
            self.trans_model = ctranslate2.Translator(self.model_path, compute_type="auto", device=self.device)
        elif "mt5" in self.model_path:
            self.mt5_prefix = self.whisper_lang.code + "2" + self.nllb_lang.code_whisper + ": "
            self.trans_tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_path, legacy=False) #requires spiece.model
            self.trans_model = transformers.MT5ForConditionalGeneration.from_pretrained(self.model_path)
            self.trans_translator = transformers.pipeline('text2text-generation', model=self.trans_model, device=self.device, tokenizer=self.trans_tokenizer)
        else: #NLLB
            self.trans_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
            self.trans_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.trans_translator = transformers.pipeline('translation', model=self.trans_model, device=self.device, tokenizer=self.trans_tokenizer, src_lang=self.nllb_whisper_lang.code, tgt_lang=self.nllb_lang.code)

    def release_vram(self):
        try:
            if torch.cuda.is_available():
                if "ct2" not in self.model_path:
                    device = torch.device("cpu")
                    self.trans_model.to(device)
                del self.trans_model
                torch.cuda.empty_cache()
                print("release vram end.")
        except Exception as e:
            print("Error release vram: " + str(e))


    def translation(self, text: str, max_length: int = 400):
        output = None
        result = None
        try:
            if "ct2" in self.model_path:
                source = self.trans_tokenizer.convert_ids_to_tokens(self.trans_tokenizer.encode(text))
                output = self.trans_model.translate_batch([source], target_prefix=[self.target_prefix])
                target = output[0].hypotheses[0][1:]
                result = self.trans_tokenizer.decode(self.trans_tokenizer.convert_tokens_to_ids(target))
            elif "mt5" in self.model_path:
                output = self.trans_translator(self.mt5_prefix + text, max_length=max_length, num_beams=4)
                result = output[0]['generated_text']
            else: #NLLB
                output = self.trans_translator(text, max_length=max_length)
                result = output[0]['translation_text']
        except Exception as e:
            print("Error translation text: " + str(e))

        return result


_MODELS = ["distilled-600M", "distilled-1.3B", "1.3B", "3.3B", 
           "ct2fast-nllb-200-distilled-1.3B-int8_float16", 
           "ct2fast-nllb-200-3.3B-int8_float16", 
           "nllb-200-3.3B-ct2-float16", "nllb-200-1.3B-ct2", "nllb-200-1.3B-ct2-int8", "nllb-200-1.3B-ct2-float16", 
           "nllb-200-distilled-1.3B-ct2", "nllb-200-distilled-1.3B-ct2-int8", "nllb-200-distilled-1.3B-ct2-float16", 
           "nllb-200-distilled-600M-ct2", "nllb-200-distilled-600M-ct2-int8", "nllb-200-distilled-600M-ct2-float16",
           "mt5-zh-ja-en-trimmed",
           "mt5-zh-ja-en-trimmed-fine-tuned-v1"]

def check_model_name(name):
    return any(allowed_name in name for allowed_name in _MODELS)

def download_model(
    model_config: ModelConfig,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
):
    """"download_model" is referenced from the "utils.py" script 
      of the "faster_whisper" project, authored by guillaumekln.
    
    Downloads a nllb-200 model from the Hugging Face Hub.

    The model is downloaded from https://huggingface.co/facebook.

    Args:
      model_config: config of the model to download (facebook/nllb-distilled-600M, 
        facebook/nllb-distilled-1.3B, facebook/nllb-1.3B, facebook/nllb-3.3B...).
      output_dir: Directory where the model should be saved. If not set, the model is saved in
        the cache directory.
      local_files_only:  If True, avoid downloading the file and return the path to the local
        cached file if it exists.
      cache_dir: Path to the folder where cached files are stored.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if not check_model_name(model_config.name):
        raise ValueError(
            "Invalid model name '%s', expected one of: %s" % (model_config.name, ", ".join(_MODELS))
        )

    repo_id = model_config.url #"facebook/nllb-200-%s" % 

    allow_patterns = [
        "config.json",
        "generation_config.json",
        "model.bin",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-00001-of-00003.bin",
        "pytorch_model-00002-of-00003.bin",
        "pytorch_model-00003-of-00003.bin",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "shared_vocabulary.txt",
        "shared_vocabulary.json",
        "special_tokens_map.json",
        "spiece.model",
    ]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
        #"tqdm_class": disabled_tqdm,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    try:
        return huggingface_hub.snapshot_download(repo_id, **kwargs)
    except (
        huggingface_hub.utils.HfHubHTTPError,
        requests.exceptions.ConnectionError,
    ) as exception:
        warnings.warn(
            "An error occured while synchronizing the model %s from the Hugging Face Hub:\n%s",
            repo_id,
            exception,
        )
        warnings.warn(
            "Trying to load the model directly from the local cache, if it exists."
        )

        kwargs["local_files_only"] = True
        return huggingface_hub.snapshot_download(repo_id, **kwargs)
