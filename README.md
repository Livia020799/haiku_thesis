# Automatic generation of Haiku using Large Language Models: comparing AI creativity with human poetry

## Overview

This repository contains the complete experimental pipeline developed for the thesis, from haiku generation to questionnaire creation, statistical analysis, and supporting literature.

The project investigates whether modern Large Language Models can generate Japanese haiku that are perceived as comparable to human-written poetry. Multiple state-of-the-art models are evaluated through a human study in which participants assess fluency, poeticness, coherence, creativity, and their ability to distinguish between AI- and human-authored haiku.

The repository is intended to provide transparency and reproducibility for the experiments presented in the thesis.

---

## Repository contents

```text
- haiku_repo/
- haiku_thesis-main/
  - README.md
  - Thesis.pdf
  - ANALYSIS/
    - gamma_haiku_questionnaire_analysis.py
    - gemini_haiku_questionnaire_analysis.py
    - gemma_haiku_questionnaire_analysis.py
    - gpt5_haiku_questionnaire_analysis.py
    - instruct4_haiku_questionnaire_analysis.py
    - llama_haiku_questionnaire_analysis.py
    - overall_graphs.py
  - HAIKU GENERATION/
    - ChatGPT5_prompt_on_smaller_models.py
    - PreProcessing.ipynb
    - Qwen.py
    - beam search (one-shot generation).py
    - beam search (one-shot or line-by-line generation).py
    - code
    - sampling with line-by-line generation.py
  - Questionnaires/
    - GPT5_AW_questionnaire.md
    - GPT5_SS_questionnaire.md
    - Gemini 2.5_AW_questionnaire.md
    - Gemini 2.5_SS_questionnaire.md
    - Gemma 2B_AW_questionnaire.md
    - Gemma 2B_SS_questionnaire.md
    - LLM-JP_AW_questionnaire.md
    - LLM-JP_SS_questionnaire.md
    - LLaMA 2_AW_questionnaire.md
    - LLaMA 2_SS_questionnaire.md
  - Research_on_haiku/
    - Haiku evaluation/
      - Aesthetic evaluation of AI_generated vs with human intervention (2023).pdf
      - Proposal of a Haiku Evaluation Method Using Large Language Model and Prompt Engineering (Feb. 2025) - Tomizawa Shunki.pdf
      - Quality_estimation_for_Japanese_Haiku_poems_using_Neural_Network (2016).pdf
    - Haiku generation/
      - Haiku Generation using Deep Neural Network.pdf
      - Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese.pdf
      - haiku generation (english).pdf
```

The repository is organised into four main components:

| Folder | Description |
|--------|-------------|
| **HAIKU GENERATION** | Scripts used to generate haiku with different prompting and decoding strategies. |
| **Questionnaires** | Google Forms questionnaires (PDF/Markdown) prepared for each evaluated model. |
| **ANALYSIS** | Scripts used to analyse participant responses and generate figures and summary statistics. |
| **Research_on_haiku** | Collection of background literature on haiku generation and evaluation. |

---

## Experimental workflow

1. Generate haiku using multiple Large Language Models.
2. Prepare evaluation questionnaires containing both human- and AI-generated haiku.
3. Collect human judgments through anonymous questionnaires.
4. Analyse responses using the scripts contained in `ANALYSIS`.
5. Produce comparative visualisations and statistical summaries.

---

## Human evaluation

Participants evaluate each haiku across several dimensions including:

- Source identification (Human vs AI)
- Confidence in the source judgment
- Theme relevance
- Fluency
- Haiku-like wording
- Poeticness
- Coherence
- Understandability
- Overall favourability
- Unexpectedness / creativity
- Optional qualitative comments

The questionnaires are available in both PDF and Markdown formats for transparency and reproducibility.

---

## Models

The repository contains material related to multiple contemporary Large Language Models, including GPT, Gemini, LLaMA, Gemma, LLM-JP and other open-source models used throughout the study.

---

## Reproducibility

The repository includes the resources necessary to reproduce the experimental workflow:

- generation scripts
- questionnaire material
- analysis scripts
- supporting literature
- thesis manuscript

Depending on the model used, access to the corresponding API or local model weights may be required.

---

## Citation

If you use this repository, please cite the accompanying thesis and any related publication.

```bibtex
@misc{haiku_thesis,
  title={Automatic generation of Haiku using Large Language Models: comparing AI creativity with human poetry},
  author={Livia Oddi},
  year={2026}
}
```

---

## Acknowledgements

This work was developed as part of a Master's thesis on Natural Language Processing and Computational Creativity. We thank all volunteers who participated in the human evaluation study.

## License

Unless otherwise stated, this repository is provided for academic and research purposes.


