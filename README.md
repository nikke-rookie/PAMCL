## Introduction

The source code of "Preference-Aware Multimodal Contrastive Learning Recommendation Method" (PAMCL). 

## Environment

The Python and package managers version are:
- Python 3.9.19
- mamba 1.5.8
- conda 24.3.0

Use conda/mamba to install the environment (**Recommend**):
```bash
conda env create -f environment.yml
```

## Datasets

Download from [google drive](https://drive.google.com/file/d/19h7L42K5m51uD8PxfZNiAWT1SOZEWrja/view?usp=sharing).

## Pre-trained models

- Vision: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- Text: [bennegeek/stella_en_1.5B_v5](https://huggingface.co/bennegeek/stella_en_1.5B_v5)

> [!warning]
> Use [bennegeek/stella_en_1.5B_v5](https://huggingface.co/bennegeek/stella_en_1.5B_v5) to generate text embeddings requires `flash-attn` which needs a CUDA version of at least 11.6, as indicated by `nvcc -V`.

## Contributions

This project is forked from the original [SELFRec](https://github.com/Coder-Yu/SELFRec). We would like to express our gratitude to Coder-Yu and his team for their outstanding work and open-source contributions.

The main changes are:
- Support multi-modal datasets
- Use Large Language Models to enhance presentations
- Update some deprecated APIs
- Improve code readability by type hint and docstrings
- Refactor some modules