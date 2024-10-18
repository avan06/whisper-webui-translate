
# Describe

The `translate` task in `Whisper` only supports translating other languages `into English`. `OpenAI` does not guarantee translations between arbitrary languages. In such cases, you can opt to use the Translation Model for translation tasks. However, it's important to note that the `Translation Model runs very slowly on CPU`, and the completion time may be twice as long as usual. It is recommended to run the Translation Model on devices with `GPUs` for better performance.  

The larger the parameters of the Translation model, the better its translation capability is expected. However, this also requires higher computational resources and slower running speed.  

The translation model is now compatible with the `Word Timestamps - Highlight Words` feature.  

~~Currently, when the `Highlight Words timestamps` option is enabled in the Whisper `Word Timestamps options`, it cannot be used simultaneously with the Translation Model. This is because Highlight Words splits the source text, and after translation, it becomes a non-word-level string.~~  


# Translation Model

The required VRAM is provided for reference and may not apply to everyone. If the model's VRAM requirement exceeds the available capacity of the system, the model will operate on the CPU, resulting in significantly longer execution times.  

[CTranslate2](https://opennmt.net/CTranslate2/guides/transformers.html) is a C++ and Python library for efficient inference with Transformer models. Models converted from CTranslate2 can run with lower resources and faster speed. Encoder-decoder models currently supported: Transformer base/big, M2M-100, NLLB, BART, mBART, Pegasus, T5, Whisper.  

## M2M100

M2M100 is a multilingual translation model introduced by Facebook AI in October 2020. It supports arbitrary translation among 101 languages. The paper is titled "`Beyond English-Centric Multilingual Machine Translation`" ([arXiv:2010.11125](https://arxiv.org/abs/2010.11125)).  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) | 418M | 1.94 GB | float32 | ≈2 GB |
| [facebook/m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) | 1.2B | 4.96 GB | float32 | ≈5 GB |
| [facebook/m2m100-12B-last-ckpt](https://huggingface.co/facebook/m2m100-12B-last-ckpt) | 12B | 47.2 GB | float32 | ≈22.1 GB (torch dtype in float16) |

## M2M100-CTranslate2

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [michaelfeil/ct2fast-m2m100_418M](https://huggingface.co/michaelfeil/ct2fast-m2m100_418M) | 418M | 970 MB | float16 | ≈0.6 GB |
| [michaelfeil/ct2fast-m2m100_1.2B](https://huggingface.co/michaelfeil/ct2fast-m2m100_1.2B) | 1.2B | 2.48 GB | float16 | ≈1.3 GB |
| [michaelfeil/ct2fast-m2m100-12B-last-ckpt](https://huggingface.co/michaelfeil/ct2fast-m2m100-12B-last-ckpt) | 12B | 23.6 GB | float16 | N/A |

## NLLB-200

NLLB-200 is a multilingual translation model introduced by Meta AI in July 2022. It supports arbitrary translation among 202 languages. The paper is titled "`No Language Left Behind: Scaling Human-Centered Machine Translation`" ([arXiv:2207.04672](https://arxiv.org/abs/2207.04672)).  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 600M | 2.46 GB | float32 | ≈2.5 GB |
| [facebook/nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | 1.3B | 5.48 GB | float32 | ≈5.9 GB |
| [facebook/nllb-200-1.3B](https://huggingface.co/facebook/nllb-200-1.3B) | 1.3B | 5.48 GB | float32 | ≈5.8 GB |
| [facebook/nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) | 3.3B | 17.58 GB | float32 | ≈13.4 GB |
| [facebook/nllb-moe-54b](https://huggingface.co/facebook/nllb-moe-54b) | 54B | 220.2 GB | float32 | N/A |

## NLLB-200-CTranslate2

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [michaelfeil/ct2fast-nllb-200-distilled-1.3B](https://huggingface.co/michaelfeil/ct2fast-nllb-200-distilled-1.3B) | 1.3B | 1.38 GB | int8_float16 | ≈1.3 GB |
| [michaelfeil/ct2fast-nllb-200-3.3B](https://huggingface.co/michaelfeil/ct2fast-nllb-200-3.3B) | 3.3B | 3.36 GB | int8_float16 | ≈3.2 GB |
| [JustFrederik/nllb-200-1.3B-ct2-int8](https://huggingface.co/JustFrederik/nllb-200-1.3B-ct2-int8) | 1.3B | 1.38 GB | int8  | ≈1.3 GB |
| [JustFrederik/nllb-200-1.3B-ct2-float16](https://huggingface.co/JustFrederik/nllb-200-1.3B-ct2-float16) | 1.3B | 2.74 GB | float16 | ≈1.3 GB |
| [JustFrederik/nllb-200-distilled-600M-ct2](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2) | 600M | 2.46 GB | float32 | ≈0.6 GB |
| [JustFrederik/nllb-200-distilled-600M-ct2-float16](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-float16) | 600M | 1.23 GB | float16 | ≈0.6 GB |
| [JustFrederik/nllb-200-distilled-600M-ct2-int8](https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8) | 600M | 623 MB | int8 | ≈0.6 GB |
| [JustFrederik/nllb-200-distilled-1.3B-ct2-float16](https://huggingface.co/JustFrederik/nllb-200-distilled-1.3B-ct2-float16) | 1.3B | 2.74 GB | float16 | ≈1.3 GB |
| [JustFrederik/nllb-200-distilled-1.3B-ct2-int8](https://huggingface.co/JustFrederik/nllb-200-distilled-1.3B-ct2-int8) | 1.3B | 1.38 GB | int8 | ≈1.3 GB |
| [JustFrederik/nllb-200-distilled-1.3B-ct2](https://huggingface.co/JustFrederik/nllb-200-distilled-1.3B-ct2) | 1.3B | 5.49 GB | float32 | ≈1.3 GB |
| [JustFrederik/nllb-200-1.3B-ct2](https://huggingface.co/JustFrederik/nllb-200-1.3B-ct2) | 1.3B | 5.49 GB | float32 | ≈1.3 GB |
| [JustFrederik/nllb-200-3.3B-ct2-float16](https://huggingface.co/JustFrederik/nllb-200-3.3B-ct2-float16) | 3.3B | 6.69 GB | float16 | ≈3.2 GB |

## MT5

mT5 is a multilingual pre-trained Text-to-Text Transformer introduced by Google Research in October 2020. It is a multilingual variant of the T5 model, pre-trained on datasets in 101 languages. Further fine-tuning is required to transform it into a translation model. The paper is titled "`mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer`" ([arXiv:2010.11934](https://arxiv.org/abs/2010.11934)).  
The 'mt5-zh-ja-en-trimmed' model is finetuned from Google's 'mt5-base' model. This model has a relatively good translation speed, but it only supports three languages: Chinese, Japanese, and English.  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [mt5-base](https://huggingface.co/google/mt5-base) | N/A | 2.33 GB | float32 | N/A |
| [K024/mt5-zh-ja-en-trimmed](https://huggingface.co/K024/mt5-zh-ja-en-trimmed) | N/A | 1.32 GB | float32 | ≈1.4 GB |
| [engmatic-earth/mt5-zh-ja-en-trimmed-fine-tuned-v1](https://huggingface.co/engmatic-earth/mt5-zh-ja-en-trimmed-fine-tuned-v1) | N/A | 1.32 GB | float32 | ≈1.4 GB |

## ALMA

ALMA is a many-to-many LLM-based translation model introduced by Haoran Xu and colleagues in September 2023. It is based on the fine-tuning of a large language model (LLaMA-2). The approach used for this model is referred to as Advanced Language Model-based trAnslator (ALMA). The paper is titled "`A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models`" ([arXiv:2309.11674](https://arxiv.org/abs/2309.11674)).  
The official support for ALMA currently includes 10 language directions: English↔German, English↔Czech, English↔Icelandic, English↔Chinese, and English↔Russian. However, the author hints that there might be surprises in other directions, so there are currently no restrictions on the languages that ALMA can be chosen for in the web UI.  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [haoranxu/ALMA-7B](https://huggingface.co/haoranxu/ALMA-7B) | 7B | 26.95 GB | float32 | ≈13.2 GB (torch dtype in float16) |
| [haoranxu/ALMA-13B](https://huggingface.co/haoranxu/ALMA-13B) | 13B | 52.07 GB | float32 | ≈25.4 GB (torch dtype in float16) |

## ALMA-GPTQ

Due to the poor support of GPTQ for CPUs, the execution time per iteration exceeds a thousand seconds when operating on a CPU. Therefore, it is strongly discouraged to operate it on CPU.  
GPTQ is a technique used to quantize the parameters of large language models into integer formats such as int8 or int4. Although the quantization process may lead to a loss in model performance, it significantly reduces both file size and the required VRAM.  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [TheBloke/ALMA-7B-GPTQ](https://huggingface.co/TheBloke/ALMA-7B-GPTQ) | 7B | 3.9 GB | 4 Bits | ≈4.3 GB |
| [TheBloke/ALMA-13B-GPTQ](https://huggingface.co/TheBloke/ALMA-13B-GPTQ) | 13B | 7.26 GB | 4 Bits | ≈8.4 GB |

## ALMA-GGUF

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.
GGUF is a file format for storing models for inference with GGML and executors based on GGML. GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML.  
[k-quants](https://github.com/ggerganov/llama.cpp/pull/1684): a series of 2-6 bit quantization methods, along with quantization mixes

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [TheBloke/ALMA-7B-GGUF-Q4_K_M](https://huggingface.co/TheBloke/ALMA-7B-GGUF) | 7B | 4.08 GB | Q4_K_M(4 Bits medium) | ≈5.3 GB |
| [TheBloke/ALMA-13B-GGUF-Q4_K_M](https://huggingface.co/TheBloke/ALMA-13B-GGUF) | 13B | 7.87 GB | Q4_K_M(4 Bits medium) | ≈9.3 GB |

## ALMA-CTranslate2

[CTranslate2](https://opennmt.net/CTranslate2/) does not currently support 4-bit quantization. Currently, it can only use int8_float16 quantization, so the file size and required VRAM will be larger than the GPTQ model quantized with 4 bits. However, it runs much faster on the CPU than GPTQ. If you plan to run ALMA in an environment without a GPU, you may consider choosing the CTranslate2 version of the ALMA model.  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [avans06/ALMA-7B-ct2-int8_float16](https://huggingface.co/avans06/ALMA-7B-ct2-int8_float16) | 7B | 6.74 GB | int8_float16 | ≈6.6 GB |
| [avans06/ALMA-13B-ct2-int8_float16](https://huggingface.co/avans06/ALMA-13B-ct2-int8_float16) | 13B | 13 GB | int8_float16 | ≈12.6 GB |

## madlad400

madlad400 is a multilingual machine translation model based on the T5 architecture introduced by Google DeepMind, Google Research in Sep 2023. It was trained on 250 billion tokens covering over 450 languages using publicly available data. The paper is titled "`MADLAD-400: A Multilingual And Document-Level Large Audited Dataset`" ([arXiv:2309.04662](https://arxiv.org/abs/2309.04662)).  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [jbochi/madlad400-3b-mt](https://huggingface.co/jbochi/madlad400-3b-mt) | 3B | 11.8 GB | float32 | ≈12 GB |
| [jbochi/madlad400-7b-mt](https://huggingface.co/jbochi/madlad400-7b-mt) | 7.2B | 33.2 GB | float32 | ≈19.7 GB (torch dtype in float16) |
| [jbochi/madlad400-7b-mt-bt](https://huggingface.co/jbochi/madlad400-7b-mt-bt) | 7.2B | 33.2 GB | float32 (finetuned on backtranslated data) | ≈19.7 GB (torch dtype in float16) |
| [jbochi/madlad400-8b-lm](https://huggingface.co/jbochi/madlad400-8b-lm) | 8B | 34.52 GB | float32 | N/A |
| [jbochi/madlad400-10b-mt](https://huggingface.co/jbochi/madlad400-10b-mt) | 10.7B | 42.86 GB | float32 | ≈24.3 GB (torch dtype in float16) |

## madlad400-CTranslate2

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [SoybeanMilk/madlad400-3b-mt-ct2-int8_float16](https://huggingface.co/SoybeanMilk/madlad400-3b-mt-ct2-int8_float16) | 3B | 2.95 GB | int8_float16 | ≈2.7 GB |
| [avans06/madlad400-7b-mt-bt-ct2-int8_float16](https://huggingface.co/avans06/madlad400-7b-mt-bt-ct2-int8_float16) | 7.2B | 8.31 GB | int8_float16 (finetuned on backtranslated data) | ≈8.5 GB |
| [SoybeanMilk/madlad400-10b-mt-ct2-int8_float16](https://huggingface.co/SoybeanMilk/madlad400-10b-mt-ct2-int8_float16) | 10.7B | 10.7 GB | int8_float16 | ≈10 GB |

## SeamlessM4T

SeamlessM4T is a collection of models designed to provide high quality translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.  

It enables multiple tasks without relying on separate models:  

Speech-to-speech translation (S2ST)  
Speech-to-text translation (S2TT)  
Text-to-speech translation (T2ST)  
Text-to-text translation (T2TT)  
Automatic speech recognition (ASR)  

[SeamlessM4T-v1](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t) introduced by Seamless Communication team from Meta AI in Aug 2023. The paper is titled "`SeamlessM4T: Massively Multilingual & Multimodal Machine Translation`"([arXiv:2308.11596](https://arxiv.org/abs/2308.11596))  
[SeamlessM4T-v2](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t_v2) introduced by Seamless Communication team from Meta AI in Dec 2023. The paper is titled "`Seamless: Multilingual Expressive and Streaming Speech Translation`"([arXiv:2312.05187](https://arxiv.org/abs/2312.05187))  

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [facebook/hf-seamless-m4t-medium](https://huggingface.co/facebook/hf-seamless-m4t-medium) | 1.2B | 4.84 GB | float32 | N/A |
| [facebook/seamless-m4t-large](https://huggingface.co/facebook/seamless-m4t-large) | 2.3B | 11.4 GB | float32 | N/A |
| [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | 2.3B | 11.4 GB (safetensors:9.24 GB) | float32 | ≈9.2 GB |

## Llama

Meta developed and released the Meta Llama 3 family of large language models (LLMs). This program modifies them through prompts to function as translation models.

| Name | Parameters | Size | type/quantize | Required VRAM |
|------|------------|------|---------------|---------------|
| [avans06/Meta-Llama-3.2-8B-Instruct-ct2-int8_float16](https://huggingface.co/avans06/Meta-Llama-3.2-8B-Instruct-ct2-int8_float16) | 8B | 8.04 GB | int8_float16 | ≈ 7.9 GB |
| [avans06/Meta-Llama-3.1-8B-Instruct-ct2-int8_float16](https://huggingface.co/avans06/Meta-Llama-3.1-8B-Instruct-ct2-int8_float16) | 8B | 8.04 GB | int8_float16 | ≈ 7.9 GB |
| [avans06/Meta-Llama-3-8B-Instruct-ct2-int8_float16](https://huggingface.co/avans06/Meta-Llama-3-8B-Instruct-ct2-int8_float16) | 8B | 8.04 GB | int8_float16 | ≈ 7.9 GB |
| [jncraton/Llama-3.2-3B-Instruct-ct2-int8](https://huggingface.co/jncraton/Llama-3.2-3B-Instruct-ct2-int8) | 3B | 3.22 GB | int8 | ≈ 3.3 GB |


# Options

## Translation - Batch Size
- transformers: batch_size  
When the pipeline will use DataLoader (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial.
- ctranslate2: max_batch_size  
The maximum batch size.

## Translation - No Repeat Ngram Size
- transformers: no_repeat_ngram_size  
Value that will be used by default in the generate method of the model for no_repeat_ngram_size. If set to int > 0, all ngrams of that size can only occur once.
- ctranslate2: no_repeat_ngram_size  
Prevent repetitions of ngrams with this size (set 0 to disable).

## Translation - Num Beams
- transformers: num_beams  
Number of beams for beam search that will be used by default in the generate method of the model. 1 means no beam search.
- ctranslate2: beam_size  
Beam size (1 for greedy search).

## Translation - Torch Dtype float16
- transformers: torch_dtype=torch.float16  
Load the float32 translation model with float16 when the system supports GPU (reducing VRAM usage, not applicable to models that have already been quantized, such as Ctranslate2, GPTQ, GGUF)

## Translation - Using Bitsandbytes
- transformers: load_in_8bit, load_in_4bit
Load the float32 translation model into mixed-8bit or 4bit precision quantized model when the system supports GPU (reducing VRAM usage, not applicable to models that have already been quantized, such as Ctranslate2, GPTQ, GGUF)