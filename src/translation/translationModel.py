import os
import warnings
import huggingface_hub
import requests
import torch

import ctranslate2
import transformers

import re

from typing import Optional
from src.config import ModelConfig
from src.translation.translationLangs import TranslationLang, get_lang_from_whisper_code

class TranslationModel:
    def __init__(
        self,
        modelConfig: ModelConfig,
        device: str = None,
        whisperLang: TranslationLang = None,
        translationLang: TranslationLang = None,
        batchSize: int = 2,
        noRepeatNgramSize: int = 3,
        numBeams: int = 2,
        downloadRoot: Optional[str] = None,
        localFilesOnly: bool = False,
        loadModel: bool = False,
    ):
        """Initializes the M2M100 / Nllb-200 / mt5 model.

        Args:
          modelConfig: Config of the model to use (distilled-600M, distilled-1.3B, 
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
          downloadRoot: Directory where the models should be saved. If not set, the models
            are saved in the standard Hugging Face cache directory.
          localFilesOnly:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
        """
        self.modelConfig = modelConfig
        self.whisperLang = whisperLang # self.translationLangWhisper = get_lang_from_whisper_code(whisperLang.code.lower() if whisperLang is not None else "en")
        self.translationLang = translationLang

        if translationLang is None:
            return
        
        self.batchSize = batchSize
        self.noRepeatNgramSize = noRepeatNgramSize
        self.numBeams = numBeams

        if os.path.isdir(modelConfig.url):
            self.modelPath = modelConfig.url
        else:
            self.modelPath = download_model(
                modelConfig,
                localFilesOnly=localFilesOnly,
                cacheDir=downloadRoot,
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda" if "ct2" in self.modelPath else "cuda:0"
            else:
                device = "cpu"

        self.device = device

        if loadModel:
            self.load_model()

    def load_model(self):
        print('\n\nLoading model: %s\n\n' % self.modelPath)
        if "ct2" in self.modelPath:
            if "nllb" in self.modelPath:
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelConfig.tokenizer_url if self.modelConfig.tokenizer_url is not None and len(self.modelConfig.tokenizer_url) > 0 else self.modelPath, src_lang=self.whisperLang.nllb.code)
                self.targetPrefix = [self.translationLang.nllb.code]
            elif "m2m100" in self.modelPath:
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelConfig.tokenizer_url if self.modelConfig.tokenizer_url is not None and len(self.modelConfig.tokenizer_url) > 0 else self.modelPath, src_lang=self.whisperLang.m2m100.code)
                self.targetPrefix = [self.transTokenizer.lang_code_to_token[self.translationLang.m2m100.code]]
            self.transModel = ctranslate2.Translator(self.modelPath, compute_type="auto", device=self.device)
        elif "mt5" in self.modelPath:
            self.mt5Prefix = self.whisperLang.whisper.code + "2" + self.translationLang.whisper.code + ": "
            self.transTokenizer = transformers.T5Tokenizer.from_pretrained(self.modelPath, legacy=False) #requires spiece.model
            self.transModel = transformers.MT5ForConditionalGeneration.from_pretrained(self.modelPath)
            self.transTranslator = transformers.pipeline('text2text-generation', model=self.transModel, device=self.device, tokenizer=self.transTokenizer)
        elif "ALMA" in self.modelPath:
            self.ALMAPrefix = "Translate this from " + self.whisperLang.whisper.code + " to " + self.translationLang.whisper.code + ":" + self.whisperLang.whisper.code + ":"
            self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelPath, use_fast=True)
            self.transModel = transformers.AutoModelForCausalLM.from_pretrained(self.modelPath, device_map="auto", trust_remote_code=False, revision="main")
            self.transTranslator = transformers.pipeline("text-generation", model=self.transModel, tokenizer=self.transTokenizer, batch_size=2, do_sample=True, temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1)
        else:
            self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelPath)
            self.transModel = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.modelPath)
            if "m2m100" in self.modelPath:
                self.transTranslator = transformers.pipeline('translation', model=self.transModel, device=self.device, tokenizer=self.transTokenizer, src_lang=self.whisperLang.m2m100.code, tgt_lang=self.translationLang.m2m100.code)
            else: #NLLB
                self.transTranslator = transformers.pipeline('translation', model=self.transModel, device=self.device, tokenizer=self.transTokenizer, src_lang=self.whisperLang.nllb.code, tgt_lang=self.translationLang.nllb.code)

    def release_vram(self):
        try:
            if torch.cuda.is_available():
                if "ct2" not in self.modelPath:
                    device = torch.device("cpu")
                    self.transModel.to(device)
                del self.transModel
                torch.cuda.empty_cache()
                print("release vram end.")
        except Exception as e:
            print("Error release vram: " + str(e))


    def translation(self, text: str, max_length: int = 400):
        output = None
        result = None
        try:
            if "ct2" in self.modelPath:
                source = self.transTokenizer.convert_ids_to_tokens(self.transTokenizer.encode(text))
                output = self.transModel.translate_batch([source], target_prefix=[self.targetPrefix], max_batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, beam_size=self.numBeams)
                target = output[0].hypotheses[0][1:]
                result = self.transTokenizer.decode(self.transTokenizer.convert_tokens_to_ids(target))
            elif "mt5" in self.modelPath:
                output = self.transTranslator(self.mt5Prefix + text, max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams) #, num_return_sequences=2
                result = output[0]['generated_text']
            elif "ALMA" in self.modelPath:
                output = self.transTranslator(self.ALMAPrefix + text + self.translationLang.whisper.code + ":", max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams)
                result = output[0]['generated_text']
                result = re.sub(rf'^(.*{self.translationLang.whisper.code}: )', '', result)  # Remove the prompt from the result
                result = re.sub(rf'^(Translate this from .* to .*:)', '', result)  # Remove the translation instruction
                return result.strip()
            else: #M2M100 & NLLB
                output = self.transTranslator(text, max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams)
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
           "m2m100_1.2B-ct2", "m2m100_418M-ct2", "m2m100-12B-ct2", 
           "m2m100_1.2B", "m2m100_418M",
           "mt5-zh-ja-en-trimmed",
           "mt5-zh-ja-en-trimmed-fine-tuned-v1",
           "ALMA-13B-GPTQ"]

def check_model_name(name):
    return any(allowed_name in name for allowed_name in _MODELS)

def download_model(
    modelConfig: ModelConfig,
    outputDir: Optional[str] = None,
    localFilesOnly: bool = False,
    cacheDir: Optional[str] = None,
):
    """"download_model" is referenced from the "utils.py" script 
      of the "faster_whisper" project, authored by guillaumekln.
    
    Downloads a nllb-200 model from the Hugging Face Hub.

    The model is downloaded from https://huggingface.co/facebook.

    Args:
      modelConfig: config of the model to download (facebook/nllb-distilled-600M, 
        facebook/nllb-distilled-1.3B, facebook/nllb-1.3B, facebook/nllb-3.3B...).
      outputDir: Directory where the model should be saved. If not set, the model is saved in
        the cache directory.
      localFilesOnly:  If True, avoid downloading the file and return the path to the local
        cached file if it exists.
      cacheDir: Path to the folder where cached files are stored.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if not check_model_name(modelConfig.name):
        raise ValueError(
            "Invalid model name '%s', expected one of: %s" % (modelConfig.name, ", ".join(_MODELS))
        )

    repoId = modelConfig.url #"facebook/nllb-200-%s" % 

    allowPatterns = [
        "config.json",
        "generation_config.json",
        "model.bin",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
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
        "vocab.json", #m2m100
        "model.safetensors",
        "quantize_config.json",
        "tokenizer.model"
    ]

    kwargs = {
        "local_files_only": localFilesOnly,
        "allow_patterns": allowPatterns,
        #"tqdm_class": disabled_tqdm,
    }

    if outputDir is not None:
        kwargs["local_dir"] = outputDir
        kwargs["local_dir_use_symlinks"] = False

    if cacheDir is not None:
        kwargs["cache_dir"] = cacheDir

    try:
        return huggingface_hub.snapshot_download(repoId, **kwargs)
    except (
        huggingface_hub.utils.HfHubHTTPError,
        requests.exceptions.ConnectionError,
    ) as exception:
        warnings.warn(
            "An error occured while synchronizing the model %s from the Hugging Face Hub:\n%s",
            repoId,
            exception,
        )
        warnings.warn(
            "Trying to load the model directly from the local cache, if it exists."
        )

        kwargs["local_files_only"] = True
        return huggingface_hub.snapshot_download(repoId, **kwargs)
