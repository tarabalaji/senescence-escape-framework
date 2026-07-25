# SenEscape

Computational framework for identifying conserved regulatory programs associated with therapy-induced senescence escape in breast cancer using cross-cell-line single-cell RNA sequencing.

## Project Overview

Chemotherapy can induce therapy-induced senescence (TIS), a state in which cancer cells stop proliferating. Although senescence is generally considered a favorable treatment outcome, a subset of senescent cells can later re-enter the cell cycle and contribute to tumor repopulation and treatment resistance.

SenEscape investigates the molecular mechanisms underlying this transition by integrating single-cell transcriptomics, machine learning, and gene regulatory network inference. The goal is to identify conserved regulatory programs associated with repopulation and prioritize candidate intervention targets through computational perturbation.

## Objectives

- Characterize transcriptional differences between control, senescent, and repopulated cells.
- Develop a cross-cell-line framework to identify conserved repopulation-associated programs.
- Infer gene regulatory networks governing therapy-induced senescence escape.
- Perform virtual perturbation to prioritize candidate regulatory targets for future experimental validation.

## Dataset

This project uses the publicly available **GSE280381** single-cell RNA sequencing dataset.

The dataset contains two breast cancer cell lines:

- MCF7
- T47D

Each cell line contains:

- Control (CTR)
- Therapy-Induced Senescence (TIS)
- Repopulated (REPOP)

with two biological replicates per condition.

Raw sequencing files are not included in this repository.

## Repository Structure

```text
data/
    raw/
    processed/
    metadata/

src/
    load_data.py
    quality_control.py
    preprocess.py
    baseline_model.py
    escape_index.py
    differential_expression.py
    regulatory_network.py
    perturbation.py
    validation.py

results/
    figures/
    tables/
    models/
    networks/

docs/
    paper_notes.md
    novelty.md
    research_log.md

tests/
```

## Setup

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the downloaded GSE280381 dataset in:

```text
data/raw/GSE280381_RAW/
```

## Planned Pipeline

1. Load and annotate raw count matrices.
2. Perform quality control and preprocessing.
3. Integrate biological replicates.
4. Analyze transcriptional differences across cell states.
5. Train cross-cell-line baseline machine learning models.
6. Construct a repopulation-associated scoring framework.
7. Infer conserved gene regulatory networks.
8. Perform virtual perturbation and prioritize candidate targets.
9. Validate findings across both cell lines.

## Current Status

Repository initialization and data preprocessing.