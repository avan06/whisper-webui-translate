# Standard Options
To transcribe or translate an audio file, you can either copy an URL from a website (all [websites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) 
supported by YT-DLP will work, including YouTube). Otherwise, upload an audio file (choose "All Files (*.*)" 
in the file selector to select any file type, including video files) or use the microphone.

For longer audio files (>10 minutes), it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option, especially if you are using the `large-v1` model. Note that `large-v2` is a lot more forgiving, but you may still want to use a VAD with a slightly higher "VAD - Max Merge Size (s)" (60 seconds or more).

## Model
Select the model that Whisper will use to transcribe the audio:

| Size      | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|-----------|------------|--------------------|--------------------|---------------|----------------|
| tiny      | 39 M       | tiny.en            | tiny               | ~1 GB         | ~32x           |
| base      | 74 M       | base.en            | base               | ~1 GB         | ~16x           |
| small     | 244 M      | small.en           | small              | ~2 GB         | ~6x            |
| medium    | 769 M      | medium.en          | medium             | ~5 GB         | ~2x            |
| large     | 1550 M     | N/A                | large              | ~10 GB        | 1x             |
| large-v2  | 1550 M     | N/A                | large              | ~10 GB        | 1x             |
| large-v3  | 1550 M     | N/A                | large              | ~10 GB        | 1x             |
| turbo     | 809 M      | N/A                | turbo              | ~6 GB         | 8x             |

## Language

Select the language, or leave it empty for Whisper to automatically detect it. 

Note that if the selected language and the language in the audio differs, Whisper may start to translate the audio to the selected 
language. For instance, if the audio is in English but you select Japaneese, the model may translate the audio to Japanese.

## Inputs
The options "URL (YouTube, etc.)", "Upload Files" or "Micriphone Input" allows you to send an audio input to the model.

### Multiple Files
Note that the UI will only process either the given URL or the upload files (including microphone) - not both. 

But you can upload multiple files either through the "Upload files" option, or as a playlist on YouTube. Each audio file will then be processed in turn, and the resulting SRT/VTT/Transcript will be made available in the "Download" section. When more than one file is processed, the UI will also generate a "All_Output" zip file containing all the text output files.

## Task
Select the task - either "transcribe" to transcribe the audio to text, or "translate" to translate it to English.

## Vad
Using a VAD will improve the timing accuracy of each transcribed line, as well as prevent Whisper getting into an infinite
loop detecting the same sentence over and over again. The downside is that this may be at a cost to text accuracy, especially
with regards to unique words or names that appear in the audio. You can compensate for this by increasing the prompt window. 

Note that English is very well handled by Whisper, and it's less susceptible to issues surrounding bad timings and infinite loops. 
So you may only need to use a VAD for other languages, such as Japanese, or when the audio is very long.

* none
  * Run whisper on the entire audio input
* silero-vad
   * Use Silero VAD to detect sections that contain speech, and run Whisper on independently on each section. Whisper is also run 
     on the gaps between each speech section, by either expanding the section up to the max merge size, or running Whisper independently 
     on the non-speech section.
* silero-vad-expand-into-gaps
   * Use Silero VAD to detect sections that contain speech, and run Whisper on independently on each section. Each spech section will be expanded
     such that they cover any adjacent non-speech sections. For instance, if an audio file of one minute contains the speech sections 
     00:00 - 00:10 (A) and 00:30 - 00:40 (B), the first section (A) will be expanded to 00:00 - 00:30, and (B) will be expanded to 00:30 - 00:60.
* silero-vad-skip-gaps
   * As above, but sections that doesn't contain speech according to Silero will be skipped. This will be slightly faster, but 
     may cause dialogue to be skipped.
* periodic-vad
   * Create sections of speech every 'VAD - Max Merge Size' seconds. This is very fast and simple, but will potentially break 
     a sentence or word in two.

## VAD - Merge Window
If set, any adjacent speech sections that are at most this number of seconds apart will be automatically merged.

## VAD - Max Merge Size (s)
Disables merging of adjacent speech sections if they are this number of seconds long.

## VAD - Process Timeout (s)
This configures the number of seconds until a process is killed due to inactivity, freeing RAM and video memory. The default value is 30 minutes.

## VAD - Padding (s)
The number of seconds (floating point) to add to the beginning and end of each speech section. Setting this to a number
larger than zero ensures that Whisper is more likely to correctly transcribe a sentence in the beginning of 
a speech section. However, this also increases the probability of Whisper assigning the wrong timestamp 
to each transcribed line. The default value is 1 second.

## VAD - Prompt Window (s)
The text of a detected line will be included as a prompt to the next speech section, if the speech section starts at most this
number of seconds after the line has finished. For instance, if a line ends at 10:00, and the next speech section starts at
10:04, the line's text will be included if the prompt window is 4 seconds or more (10:04 - 10:00 = 4 seconds).

Note that detected lines in gaps between speech sections will not be included in the prompt 
(if silero-vad or silero-vad-expand-into-gaps) is used.

## Diarization

If checked, Pyannote will be used to detect speakers in the audio, and label them as (SPEAKER 00), (SPEAKER 01), etc. 

This requires a HuggingFace API key to function, which can be supplied with the `--auth_token` command line option for the CLI,
set in the `config.json5` file for the GUI, or provided via the `HF_ACCESS_TOKEN` environment variable.

## Diarization - Speakers

The number of speakers to detect. If set to 0, Pyannote will attempt to detect the number of speakers automatically.

# Command Line Options

Both `app.py` and `cli.py` also accept command line options, such as the ability to enable parallel execution on multiple
CPU/GPU cores, the default model name/VAD and so on. Consult the README in the root folder for more information.

# Additional Options

In addition to the above, there's also a "Full" options interface that allows you to set all the options available in the Whisper 
model. The options are as follows:

## Initial Prompt
Optional text to provide as a prompt for the first 30 seconds window. Whisper will attempt to use this as a starting point for the transcription, but you can
also get creative and specify a style or format for the output of the transcription.

For instance, if you use the prompt "hello how is it going always use lowercase no punctuation goodbye one two three start stop i you me they", Whisper will 
be biased to output lower capital letters and no punctuation, and may also be biased to output the words in the prompt more often.

## Temperature
The temperature to use when sampling. Default is 0 (zero). A higher temperature will result in more random output, while a lower temperature will be more deterministic.

## Best Of - Non-zero temperature
The number of candidates to sample from when sampling with non-zero temperature. Default is 5.

## Beam Size - Zero temperature
The number of beams to use in beam search when sampling with zero temperature. Default is 5.

## Patience - Zero temperature
The patience value to use in beam search when sampling with zero temperature. As in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search.

## Length Penalty - Any temperature
The token length penalty coefficient (alpha) to use when sampling with any temperature. As in https://arxiv.org/abs/1609.08144, uses simple length normalization by default.

## Suppress Tokens - Comma-separated list of token IDs
A comma-separated list of token IDs to suppress during sampling. The default value of "-1" will suppress most special characters except common punctuations.

## Condition on previous text
If True, provide the previous output of the model as a prompt for the next window. Disabling this may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop.

## FP16
Whether to perform inference in fp16. True by default.

## Temperature increment on fallback
The temperature to increase when falling back when the decoding fails to meet either of the thresholds below. Default is 0.2.

## Compression ratio threshold
If the gzip compression ratio is higher than this value, treat the decoding as failed. Default is 2.4.

## Logprob threshold
If the average log probability is lower than this value, treat the decoding as failed. Default is -1.0.

## No speech threshold
If the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence. Default is 0.6.

## Diarization - Min Speakers

The minimum number of speakers for Pyannote to detect.

## Diarization - Max Speakers

The maximum number of speakers for Pyannote to detect.

## Repetition Penalty
- ctranslate2: repetition_penalty  
This parameter only takes effect in [faster-whisper (ctranslate2)](https://github.com/SYSTRAN/faster-whisper/issues/478).
Penalty applied to the score of previously generated tokens (set > 1 to penalize).

## No Repeat Ngram Size
- ctranslate2: no_repeat_ngram_size  
This parameter only takes effect in [faster-whisper (ctranslate2)](https://github.com/SYSTRAN/faster-whisper/issues/478).
Prevent repetitions of ngrams with this size (set 0 to disable).

## Whisper Filter options
**This is an experimental feature and may potentially filter out correct transcription results.**  

when enabled, can effectively improve the whisper hallucination, especially for the large-v3 version of the whisper model.  

Observations for transcriptions:
1. duration: calculated by subtracting start from end, it might indicate hallucinated results when inversely proportional to text length.
1. segment_last: the last result for each segment during VAD transcription has a certain probability of being a hallucinated result.
1. avg_logprob: average log probability, ranging from logprob_threshold (default: -1) to 0, is better when a larger value. A value lower than -0.9 might suggest a poor result.
1. compression_ratio: gzip compression ratio, ranging from 0 to compression_ratio_threshold (default: 2.4), a higher positive value is preferable. If it is lower than 0.9, it might indicate suboptimal results.
1. no_speech_prob: no_speech(<|nospeech|> token) probability, ranging from 0 to no_speech_threshold (default: 0.6), a smaller positive value is preferable. If it exceeds 0.1, it might suggest suboptimal results.

Four sets of filtering conditions have now been established, utilizing text length, duration length, as well as the avg_logprob, compression_ratio, and no_speech_prob parameters returned by Whisper.
1. avg_logprob < -0.9
1. (durationLen < 1.5 || segment_last), textLen > 5, avg_logprob < -0.4, no_speech_prob > 0.5
1. (durationLen < 1.5 || segment_last), textLen > 5, avg_logprob < -0.4, no_speech_prob > 0.07, compression_ratio < 0.9
1. (durationLen < 1.5 || segment_last), compression_ratio < 0.9, no_speech_prob > 0.1

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