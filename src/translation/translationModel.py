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
        torchDtypeFloat16: bool = True,
        usingBitsandbytes: str = None,
        downloadRoot: Optional[str] = None,
        localFilesOnly: bool = False,
        loadModel: bool = False,
    ):
        """Initializes the M2M100 / Nllb-200 / mt5 / ALMA / madlad400 translation model.

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
            self.modelPath = modelConfig.url if getattr(modelConfig, "model_file", None) is not None else download_model(
                modelConfig,
                localFilesOnly=localFilesOnly,
                cacheDir=downloadRoot,
            )

        if device is None:
            self.totalVram = 0
            if torch.cuda.is_available():
                try:
                    deviceId = torch.cuda.current_device()
                    self.totalVram = torch.cuda.get_device_properties(deviceId).total_memory/(1024*1024*1024)
                except Exception as e:
                    print(traceback.format_exc())
                    print("Error detect vram: " + str(e))
                device = "cuda" if "ct2" in self.modelPath else "cuda:0"
            else:
                device = "cpu"
                torchDtypeFloat16 = False

        self.device = device
        self.torchDtypeFloat16 = torchDtypeFloat16
        self.usingBitsandbytes = usingBitsandbytes

        if loadModel:
            self.load_model()

    def load_model(self):
        """
        [transformers.BitsAndBytesConfig]
        load_in_8bit (bool, optional, defaults to False)
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (bool, optional, defaults to False)
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.
        llm_int8_enable_fp32_cpu_offload (bool, optional, defaults to False)
            This flag is used for advanced use cases and users that are aware of this feature. 
            If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. 
            This is useful for offloading large models such as google/flan-t5-xxl. Note that the int8 operations will not be run on CPU.
        bnb_4bit_compute_dtype (torch.dtype or str, optional, defaults to torch.float32)
            This sets the computational type which might be different than the input time. 
            For example, inputs might be fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (str, optional, defaults to "fp4")
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by fp4 or nf4.
        bnb_4bit_use_double_quant (bool, optional, defaults to False)
            This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.
        
        [from_pretrained]
        low_cpu_mem_usage(bool, optional):
            Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. This is an experimental feature and a subject to change at any moment.
        
        torch_dtype (str or torch.dtype, optional):
            Override the default torch.dtype and load the model under a specific dtype. The different options are:
            1. torch.float16 or torch.bfloat16 or torch.float: load in a specified dtype, ignoring the model’s config.torch_dtype if one exists. 
               If not specified the model will get loaded in torch.float (fp32).
            2. "auto" - A torch_dtype entry in the config.json file of the model will be attempted to be used. 
               If this entry isn’t found then next check the dtype of the first weight in the checkpoint that’s of a floating point type and use that as dtype. 
               This will load the model using the dtype it was saved in at the end of the training. It can’t be used as an indicator of how the model was trained. 
               Since it could be trained in one of half precision dtypes, but saved in fp32.
            For some models the dtype they were trained in is unknown - you may try to check the model’s paper or reach out to the authors and 
            ask them to add this information to the model’s card and to insert the torch_dtype entry in config.json on the hub.
            
        device_map (str or Dict[str, Union[int, str, torch.device]] or int or torch.device, optional):
            A map that specifies where each submodule should go. It doesn’t need to be refined to each parameter/buffer name, 
            once a given module name is inside, every submodule of it will be sent to the same device. 
            If we only pass the device (e.g., "cpu", "cuda:1", "mps", or a GPU ordinal rank like 1) on which the model will be allocated, 
            the device map will map the entire model to this device. Passing device_map = 0 means put the whole model on GPU 0.
            To have Accelerate compute the most optimized device_map automatically, set device_map="auto". For more information about each option see designing a device map.
        
        load_in_8bit (bool, optional, defaults to False)
            If True, will convert the loaded model into mixed-8bit quantized model. To use this feature please install bitsandbytes (pip install -U bitsandbytes).
        load_in_4bit (bool, optional, defaults to False)
            If True, will convert the loaded model into 4bit precision quantized model. To use this feature install the latest version of bitsandbytes (pip install -U bitsandbytes).
        
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
            
        [ctransformers]
        gpu_layers
            means number of layers to run on GPU. Depending on how much GPU memory is available you can increase gpu_layers. Start with a larger value gpu_layers=100 and if it runs out of memory, try smaller values.
            To run some of the model layers on GPU, set the `gpu_layers` parameter
            https://github.com/marella/ctransformers/issues/68
        """
        try:
            print('\n\nLoading model: %s\n\n' % self.modelPath)
            kwargsTokenizer = {}
            kwargsModel = {}
            kwargsPipeline = {}

            if not any(name in self.modelPath for name in ["ct2", "GGUF", "GPTQ"]):
                kwargsModel["torch_dtype"] = torch.float16 if self.torchDtypeFloat16 else "auto"

            if "GPTQ" in self.modelPath:
                kwargsModel.update({"device_map": "auto"})
            elif "ct2" in self.modelPath:
                kwargsModel.update({"device": self.device})
            elif "GGUF" in self.modelPath:
                pass
            elif self.usingBitsandbytes == None:
                    kwargsPipeline.update({"device": self.device})
            elif self.usingBitsandbytes == "int8":
                kwargsModel.update({"load_in_8bit": True, "llm_int8_enable_fp32_cpu_offload": True})
            elif self.usingBitsandbytes == "int4":
                kwargsModel.update({"load_in_4bit": True, "llm_int8_enable_fp32_cpu_offload": True, 
                                    "bnb_4bit_use_double_quant": True, 
                                    "bnb_4bit_quant_type": "nf4", 
                                    "bnb_4bit_compute_dtype": torch.bfloat16})

            if not any(name in self.modelPath for name in ["ct2", "GGUF"]):    
                kwargsModel.update({"pretrained_model_name_or_path": self.modelPath, "low_cpu_mem_usage": True})            

            if "ct2" in self.modelPath:
                kwargsTokenizer.update({"pretrained_model_name_or_path": self.modelConfig.tokenizer_url if self.modelConfig.tokenizer_url is not None and len(self.modelConfig.tokenizer_url) > 0 else self.modelPath})
                kwargsModel.update({"model_path": self.modelPath, "compute_type": "auto"})
                if "ALMA" in self.modelPath:
                    self.ALMAPrefix = "Translate this from " + self.whisperLang.whisper.names[0] + " to " + self.translationLang.whisper.names[0] + ":\n" + self.whisperLang.whisper.names[0] + ": "
                    self.transModel = ctranslate2.Generator(**kwargsModel)
                else:
                    if "nllb" in self.modelPath:
                        kwargsTokenizer.update({"src_lang": self.whisperLang.nllb.code})
                        self.targetPrefix = [self.translationLang.nllb.code]
                    elif "m2m100" in self.modelPath:
                        kwargsTokenizer.update({"src_lang": self.whisperLang.m2m100.code})
                    elif "madlad400" in self.modelPath:
                        kwargsTokenizer.update({"src_lang": self.whisperLang.m2m100.code})
                        self.madlad400Prefix = "<2" + self.translationLang.whisper.code + "> "
                    self.transModel = ctranslate2.Translator(**kwargsModel)
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(**kwargsTokenizer)
                if "m2m100" in self.modelPath:
                    self.targetPrefix = [self.transTokenizer.lang_code_to_token[self.translationLang.m2m100.code]]
            elif "mt5" in self.modelPath:
                self.mt5Prefix = self.whisperLang.whisper.code + "2" + self.translationLang.whisper.code + ": "
                kwargsTokenizer.update({"pretrained_model_name_or_path": self.modelPath, "legacy": False})
                self.transTokenizer = transformers.T5Tokenizer.from_pretrained(**kwargsTokenizer)
                self.transModel = transformers.MT5ForConditionalGeneration.from_pretrained(**kwargsModel)
                kwargsPipeline.update({"task": "text2text-generation", "model": self.transModel, "tokenizer": self.transTokenizer})
            elif "ALMA" in self.modelPath:
                self.ALMAPrefix = "Translate this from " + self.whisperLang.whisper.names[0] + " to " + self.translationLang.whisper.names[0] + ":\n" + self.whisperLang.whisper.names[0] + ": "
                if "GGUF" in self.modelPath:
                    kwargsTokenizer.update({"pretrained_model_name_or_path": self.modelConfig.tokenizer_url})
                    kwargsModel.update({"model_path_or_repo_id": self.modelPath, "hf": True, "model_file": self.modelConfig.model_file, "model_type": "llama"})
                    if self.totalVram > 2:
                        kwargsModel.update({"gpu_layers":int(self.totalVram*7)})
                    import ctransformers
                    self.transModel = ctransformers.AutoModelForCausalLM.from_pretrained(**kwargsModel)
                else:
                    kwargsTokenizer.update({"pretrained_model_name_or_path": self.modelPath, "use_fast": True})
                    if "GPTQ" in self.modelPath:
                        kwargsModel.update({"trust_remote_code": False, "revision": self.modelConfig.revision})
                        if self.device == "cpu":
                            # Due to the poor support of GPTQ for CPUs, Therefore, it is strongly discouraged to operate it on CPU.  
                            # set torch_dtype=torch.float32 to prevent the occurrence of the exception "addmm_impl_cpu_ not implemented for 'Half'."
                            transModelConfig = transformers.AutoConfig.from_pretrained(self.modelPath)
                            transModelConfig.quantization_config["use_exllama"] = False
                            kwargsModel.update({"config": transModelConfig})
                    self.transModel = transformers.AutoModelForCausalLM.from_pretrained(**kwargsModel)
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(**kwargsTokenizer)
                kwargsPipeline.update({"task": "text-generation", "model": self.transModel, "tokenizer": self.transTokenizer, "do_sample": True, "temperature": 0.7, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.1})
            elif "madlad400" in self.modelPath:
                self.madlad400Prefix = "<2" + self.translationLang.whisper.code + "> "
                kwargsTokenizer.update({"pretrained_model_name_or_path": self.modelPath, "legacy": False})
                self.transTokenizer = transformers.T5Tokenizer.from_pretrained(**kwargsTokenizer)
                self.transModel = transformers.T5ForConditionalGeneration.from_pretrained(**kwargsModel)
                kwargsPipeline.update({"task": "text2text-generation", "model": self.transModel, "tokenizer": self.transTokenizer})
            else:
                kwargsTokenizer.update({"pretrained_model_name_or_path": self.modelPath})
                self.transTokenizer = transformers.AutoTokenizer.from_pretrained(**kwargsTokenizer)
                self.transModel = transformers.AutoModelForSeq2SeqLM.from_pretrained(**kwargsModel)
                kwargsPipeline.update({"task": "translation", "model": self.transModel, "tokenizer": self.transTokenizer})
                if "m2m100" in self.modelPath:
                    kwargsPipeline.update({"src_lang": self.whisperLang.m2m100.code, "tgt_lang": self.translationLang.m2m100.code})
                else: #NLLB
                    kwargsPipeline.update({"src_lang": self.whisperLang.nllb.code, "tgt_lang": self.translationLang.nllb.code})
            if "ct2" not in self.modelPath:
                self.transTranslator = transformers.pipeline(**kwargsPipeline)
        except Exception as e:
            self.release_vram()
            raise e
            

    def release_vram(self):
        try:
            if torch.cuda.is_available():
                if "ct2" not in self.modelPath:
                    try:
                        if getattr(self, "transModel", None) is not None:
                            device = torch.device("cpu")
                            self.transModel.to(device)
                    except Exception as e:
                        print(traceback.format_exc())
                        print("\tself.transModel.to cpu, error: " + str(e))
                    if getattr(self, "transTranslator", None) is not None:
                        del self.transTranslator
                if "ct2" in self.modelPath:
                    if getattr(self, "transModel", None) is not None and getattr(self.transModel, "unload_model", None) is not None:
                        self.transModel.unload_model()
                    
                if getattr(self, "transTokenizer", None) is not None:
                    del self.transTokenizer
                if getattr(self, "transModel", None) is not None:
                    del self.transModel
                import gc
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(traceback.format_exc())
                    print("\tcuda empty cache, error: " + str(e))
                print("release vram end.")
        except Exception as e:
            print(traceback.format_exc())
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
                elif "madlad400" in self.modelPath:
                    source = self.transTokenizer.convert_ids_to_tokens(self.transTokenizer.encode(self.madlad400Prefix + text))
                    output = self.transModel.translate_batch([source], max_batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, beam_size=self.numBeams)
                    target = output[0].hypotheses[0]
                    result = self.transTokenizer.decode(self.transTokenizer.convert_tokens_to_ids(target))
            elif "mt5" in self.modelPath:
                output = self.transTranslator(self.mt5Prefix + text, max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams) #, num_return_sequences=2
                result = output[0]['generated_text']
            elif "ALMA" in self.modelPath:
                if "GPTQ" in self.modelPath:
                    output = self.transTranslator(self.ALMAPrefix + text + "\n" + self.translationLang.whisper.names[0] + ": ", max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams, return_full_text=False)
                elif "GGUF" in self.modelPath:
                    output = self.transTranslator(self.ALMAPrefix + text + "\n" + self.translationLang.whisper.names[0] + ": ", max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams, return_full_text=False)
                else:
                    output = self.transTranslator(self.ALMAPrefix + text + "\n" + self.translationLang.whisper.names[0] + ": ", max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams, return_full_text=False)
                    
                result = output[0]['generated_text']
            elif "madlad400" in self.modelPath:
                output = self.transTranslator(self.madlad400Prefix + text, max_length=max_length, batch_size=self.batchSize, no_repeat_ngram_size=self.noRepeatNgramSize, num_beams=self.numBeams) #, num_return_sequences=2
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
           "ALMA",
           "madlad400"]

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
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "shared_vocabulary.txt",
        "shared_vocabulary.json",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json", #m2m100
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "quantize_config.json",
        "tokenizer.model",
        "vocabulary.json"
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
