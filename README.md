# AutoEIT

![Google Summer of Code x HumanAI Foundation](AutoEIT_Audio-to-text_transcription_for_second_additional_language_learner_data_Mykyta_Hrebeniuk/docs/images/blog_banner.png)
*Image credits: Google Summer of Code and HumanAI*

The **Elicited Imitation Task (EIT)** is a widely used sentence-repetition test: a learner hears a sentence in the target language and repeats it back, and how closely the repetition matches the original serves as a proxy for their overall language proficiency. It is a valuable source of data for second/additional language (L2A) research, but turning raw EIT recordings into research-ready results is currently a slow, labor-intensive process — audio has to be manually transcribed, and transcripts have to be scored against a rubric by trained human raters. Neither step scales well to large datasets, and manual rubric scoring in particular is inconsistent even among experienced raters.

To address this, the HumanAI Foundation launched **AutoEIT**, where contributors build the two halves of an automated EIT pipeline for L2 Spanish: converting learner audio into accurate transcriptions, and scoring those transcriptions against the EIT rubric in a consistent, reproducible way.

## The problem

Commercial speech-to-text tools are trained on native or near-native speech and struggle with learner language, which often features transfer effects, phonological variation, disfluencies, and only partially accurate sentence repetitions. Meanwhile, existing automated scoring tools — including large language models — tend to apply the EIT rubric inconsistently, awarding different scores to the same sentence across sessions, which makes them unsuitable for research that depends on standardized, replicable scoring.

## Projects

1. **Audio-to-text transcription for second/additional language learner data**
   ([`AutoEIT_Audio-to-text_transcription_for_second_additional_language_learner_data_Mykyta_Hrebeniuk/`](AutoEIT_Audio-to-text_transcription_for_second_additional_language_learner_data_Mykyta_Hrebeniuk/))
   A segmentation and multi-model ASR transcription pipeline that converts raw EIT recordings into accurate text transcripts, evaluated against human raters across twelve ASR systems. See that folder's [README](AutoEIT_Audio-to-text_transcription_for_second_additional_language_learner_data_Mykyta_Hrebeniuk/README.md) and [final report](AutoEIT_Audio-to-text_transcription_for_second_additional_language_learner_data_Mykyta_Hrebeniuk/docs/GSOC_FINAL_REPORT.md) for the full writeup.

2. **Automated scoring system for elicited imitation task responses**
   ([`AutoEIT_Automated_scoring_system_for_elicited_imitation_task_responses_Shreya_Khanna/`](AutoEIT_Automated_scoring_system_for_elicited_imitation_task_responses_Shreya_Khanna/))
   A rule-driven, reproducible scoring engine that applies the EIT rubric to transcriptions produced by the pipeline above, targeting 90% sentence-level agreement with experienced human raters.

Each project is self-contained in its own folder, with its own documentation, code, and dependencies. For details on either, see the linked folder above.

Both projects were built as [Google Summer of Code](https://summerofcode.withgoogle.com/) contributions with [HumanAI](https://humanai.foundation/).

## GSoC final reports

- **Audio-to-text transcription:** [`GSOC_FINAL_REPORT.md`](AutoEIT_Audio-to-text_transcription_for_second_additional_language_learner_data_Mykyta_Hrebeniuk/docs/GSOC_FINAL_REPORT.md) — goals, results, best models, and lessons learned.
- **Automated scoring system:** final report forthcoming.
