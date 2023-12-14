import os
import warnings
import huggingface_hub
import requests
import torch
import ctranslate2
import transformers
import traceback

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
        """
        [from_pretrained]
        low_cpu_mem_usage(bool, optional)
        Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. This is an experimental feature and a subject to change at any moment.
        
        [transformers.AutoTokenizer.from_pretrained]
        use_fast (bool, optional, defaults to True):
            Use a fast Rust-based tokenizer if it is supported for a given model. 
            If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        
        [transformers.AutoModelForCausalLM.from_pretrained]
        device_map (str or Dict[str, Union[int, str, torch.device], optional):
            Sent directly as model_kwargs (just a simpler shortcut). When accelerate library is present, 
            set device_map="auto" to compute the most optimized device_map automatically.
        revision (str, optional, defaults to "main"):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, 
            since we use a git-based system for storing models and other artifacts on huggingface.co, 
            so revision can be any identifier allowed by git.
        code_revision (str, optional, defaults to "main")
            The specific revision to use for the code on the Hub, if the code leaves in a different repository than the rest of the model. 
            It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, 
            so revision can be any identifier allowed by git.
        trust_remote_code (bool, optional, defaults to False):
            Whether or not to allow for custom models defined on the Hub in their own modeling files. 
            This option should only be set to True for repositories you trust and in which you have read the code, 
            as it will execute code present on the Hub on your local machine.
        
        [transformers.pipeline "text-generation"]
        do_sample:
            if set to True, this parameter enables decoding strategies such as multinomial sampling, 
            beam-search multinomial sampling, Top-K sampling and Top-p sampling. 
            All these strategies select the next token from the probability distribution 
            over the entire vocabulary with various strategy-specific adjustments.
        temperature (float, optional, defaults to 1.0):
            The value used to modulate the next token probabilities.
        top_k (int, optional, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float, optional, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        repetition_penalty (float, optional, defaults to 1.0)
            The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
            
        [transformers.GPTQConfig]
        use_exllama (bool, optional):
            Whether to use exllama backend. Defaults to True if unset. Only works with bits = 4.
        
        [ExLlama]
            ExLlama is a Python/C++/CUDA implementation of the Llama model that is designed for faster inference with 4-bit GPTQ weights (check out these benchmarks). 
            The ExLlama kernel is activated by default when you create a [GPTQConfig] object. 
            To boost inference speed even further, use the ExLlamaV2 kernels by configuring the exllama_config parameter.
            The ExLlama kernels are only supported when the entire model is on the GPU. 
            If you're doing inference on a CPU with AutoGPTQ (version > 0.4.2), then you'll need to disable the ExLlama kernel. 
            This overwrites the attributes related to the ExLlama kernels in the quantization config of the config.json file.
            https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization.md#exllama
        """
        try:
            print('\n\nLoading model: %s\n\n' % self.modelPath)
            if "ct2" in self.modelPath:
                if any(name in self.modelPath for name in ["nllb", "m2m100"]):
                    if "nllb" in self.modelPath:
                        self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelConfig.tokenizer_url if self.modelConfig.tokenizer_url is not None and len(self.modelConfig.tokenizer_url) > 0 else self.modelPath, src_lang=self.whisperLang.nllb.code)
                        self.targetPrefix = [self.translationLang.nllb.code]
                    elif "m2m100" in self.modelPath:
                        self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelConfig.tokenizer_url if self.modelConfig.tokenizer_url is not None and len(self.modelConfig.tokenizer_url) > 0 else self.modelPath, src_lang=self.whisperLang.m2m100.code)
                        self.targetPrefix = [self.transTokenizer.lang_code_to_token[self.translationLang.m2m100.code]]
                    self.transModel = ctranslate2.Translator(self.modelPath, compute_type="auto", device=self.device)
                elif "ALMA" in self.modelPath:
                    self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelConfig.tokenizer_url if self.modelConfig.tokenizer_url is not None and len(self.modelConfig.tokenizer_url) > 0 else self.modelPath)
                    self.ALMAPrefix = "Translate this from " + self.whisperLang.whisper.names[0] + " to " + self.translationLang.whisper.names[0] + ":\n" + self.whisperLang.whisper.names[0] + ": "
                    self.transModel = ctranslate2.Generator(self.modelPath, device=self.device)
            elif "mt5" in self.modelPath:
                self.mt5Prefix = self.whisperLang.whisper.code + "2" + self.translationLang.whisper.code + ": "
                self.transTokenizer = transformers.T5Tokenizer.from_pretrained(self.modelPath, legacy=False) #requires spiece.model
                self.transModel = transformers.MT5ForConditionalGeneration.from_pretrained(self.modelPath, low_cpu_mem_usage=True)
                self.transTranslator = transformers.pipeline('text2text-generation', model=self.transModel, device=self.device, tokenizer=self.transTokenizer)
            elif "ALMA" in self.modelPath:
                self.ALMAPrefix = "Translate this from " + self.whisperLang.whisper.names[0] + " to " + self.translationLang.whisper.names[0] + ":\n" + self.whisperLang.whisper.names[0] + ": "
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelPath, use_fast=True)
                transModelConfig = transformers.AutoConfig.from_pretrained(self.modelPath)
                if self.device == "cpu":
                    # ALMA is an excellent translation model, but it is strongly discouraged to operate it on CPU.
                    # set torch_dtype=torch.float32 to prevent the occurrence of the exception "addmm_impl_cpu_ not implemented for 'Half'."
                    transModelConfig.quantization_config["use_exllama"] = False
                    self.transModel = transformers.AutoModelForCausalLM.from_pretrained(self.modelPath, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=False, revision=self.modelConfig.revision, config=transModelConfig, torch_dtype=torch.float32)
                else:
                    # transModelConfig.quantization_config["exllama_config"] = {"version":2} # After configuring to use ExLlamaV2, VRAM cannot be effectively released, which may be an issue. Temporarily not adopting the V2 version.
                    self.transModel = transformers.AutoModelForCausalLM.from_pretrained(self.modelPath, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=False, revision=self.modelConfig.revision)
                self.transTranslator = transformers.pipeline("text-generation", model=self.transModel, tokenizer=self.transTokenizer, do_sample=True, temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.1)
            else:
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(self.modelPath)
                self.transModel = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.modelPath)
                if "m2m100" in self.modelPath:
                    self.transTranslator = transformers.pipeline('translation', model=self.transModel, device=self.device, tokenizer=self.transTokenizer, src_lang=self.whisperLang.m2m100.code, tgt_lang=self.translationLang.m2m100.code)
                else: #NLLB
                    self.transTranslator = transformers.pipeline('translation', model=self.transModel, device=self.device, tokenizer=self.transTokenizer, src_lang=self.whisperLang.nllb.code, tgt_lang=self.translationLang.nllb.code)

        except Exception as e:
            print(traceback.format_exc())
            self.release_vram()

    def release_vram(self):
        try:
            if torch.cuda.is_available():
                if "ct2" not in self.modelPath:
                    try:
                        device = torch.device("cpu")
                        self.transModel.to(device)
                    except Exception as e:
                        print(traceback.format_exc())
                        print("\tself.transModel.to cpu, error: " + str(e))
                    del self.transTranslator
                del self.transTokenizer
                del self.transModel
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(traceback.format_exc())
                    print("\tcuda empty cache, error: " + str(e))
                import gc
                gc.collect()
                print("release vram end.")
        except Exception as e:
            print("Error release vram: " + str(e))


    def translation(self, text: str, max_length: int = 400):
        """
        [ctranslate2]
        max_batch_size:
            The maximum batch size. If the number of inputs is greater than max_batch_size, 
            the inputs are sorted by length and split by chunks of max_batch_size examples 
            so that the number of padding positions is minimized.
        no_repeat_ngram_size:
            Prevent repetitions of ngrams with this size (set 0 to disable).
        beam_size:
            Beam size (1 for greedy search).
        
        [ctranslate2.Generator.generate_batch]
        sampling_temperature:
            Sampling temperature to generate more random samples.
        sampling_topk:
            Randomly sample predictions from the top K candidates.
        sampling_topp:
            Keep the most probable tokens whose cumulative probability exceeds this value.
        repetition_penalty:
            Penalty applied to the score of previously generated tokens (set > 1 to penalize).
        include_prompt_in_result:
            Include the start_tokens in the result.
            If include_prompt_in_result is True (the default), the decoding loop is constrained to generate the start tokens that are then included in the result.
            If include_prompt_in_result is False, the start tokens are forwarded in the decoder at once to initialize its state (i.e. the KV cache for Transformer models).
            For variable-length inputs, only the tokens up to the minimum length in the batch are forwarded at once. The remaining tokens are generated in the decoding loop with constrained decoding.

        [transformers.TextGenerationPipeline.__call__]
        return_full_text (bool, optional, defaults to True):
            If set to False only added text is returned, otherwise the full text is returned. Only meaningful if return_text is set to True.
        """
        output = None
        result = None
        try:
            if "ct2" in self.modelPath:
                if any(name in self.modelPath for name in ["nllb", "m2m100"]):
                    source = self.transTokenizer.convert_ids_to_tokens(self.transTokenizer.encode(text))
                    output = self.transModel.translate_batch([source], target_prefix=[self.targetPrefix], max_batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, beam_size=self.numBeams)
                    target = output[0].hypotheses[0][1:]
                    result = self.transTokenizer.decode(self.transTokenizer.convert_tokens_to_ids(target))
                elif "ALMA" in self.modelPath:
                    source = self.transTokenizer.convert_ids_to_tokens(self.transTokenizer.encode(self.ALMAPrefix + text + "\n" + self.translationLang.whisper.names[0] + ": "))
                    output = self.transModel.generate_batch([source], max_length=max_length, max_batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, beam_size=self.numBeams, sampling_temperature=0.7, sampling_topp=0.9, repetition_penalty=1.1, include_prompt_in_result=False) #, sampling_topk=40
                    target = output[0]
                    result = self.transTokenizer.decode(target.sequences_ids[0])
            elif "mt5" in self.modelPath:
                output = self.transTranslator(self.mt5Prefix + text, max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams) #, num_return_sequences=2
                result = output[0]['generated_text']
            elif "ALMA" in self.modelPath:
                output = self.transTranslator(self.ALMAPrefix + text + "\n" + self.translationLang.whisper.names[0] + ": ", max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams, return_full_text=False)
                result = output[0]['generated_text']
            else: #M2M100 & NLLB
                output = self.transTranslator(text, max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams)
                result = output[0]['translation_text']
        except Exception as e:
            print(traceback.format_exc())
            print("Error translation text: " + str(e))

        return result


_MODELS = ["nllb-200", 
           "m2m100",
           "mt5",
           "ALMA"]

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
    
    if modelConfig.revision is not None:
        kwargs["revision"] = modelConfig.revision

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
