# Multi-Agent Radiology Report Generation

This repository contains the original five-agent radiology report-generation
pipeline and a self-contained educational notebook developed as executable
companion code for a book chapter.

[Open the chapter demo in Google Colab](https://colab.research.google.com/github/SoulEmol/multi-agent-rrg/blob/main/chapter-demo/multi_agent_rrg_demo.ipynb)

## Chapter companion

The recommended entry point is
[`chapter-demo/multi_agent_rrg_demo.ipynb`](chapter-demo/multi_agent_rrg_demo.ipynb).
It is a single-file notebook that can run from top to bottom on a Google Colab
T4 GPU without an API key or access to a controlled clinical dataset.

The notebook:

1. installs its dependencies;
2. sets deterministic seeds;
3. downloads two openly licensed teaching radiographs;
4. loads openly downloadable models and an embedded synthetic retrieval set;
5. runs all five agents;
6. displays every intermediate output;
7. performs lightweight automatic software checks; and
8. demonstrates how the pipeline handles conflicting retrieval and visual evidence.

## Five-agent pipeline

| Agent | Role | Demonstration component |
| --- | --- | --- |
| Retrieval Agent | Retrieves related report text for a query image | BiomedCLIP |
| Draft Agent | Produces a retrieval-grounded preliminary report | Qwen2.5-1.5B-Instruct |
| Refiner Agent | Audits the draft against retrieved evidence | Qwen2.5-1.5B-Instruct |
| Vision Agent | Independently describes visible image evidence | SmolVLM-500M-Instruct |
| Synthesis Agent | Reconciles the text and visual evidence | Qwen2.5-1.5B-Instruct |

The model-specific logic is isolated behind `retrieval_agent()`,
`generate_text()`, and `generate_vision()`. These models are demonstration
defaults and can be replaced with other local models, open checkpoints, or
approved APIs that preserve the same input-output interfaces.

## Quick start

1. Open the Colab link above or upload
   `chapter-demo/multi_agent_rrg_demo.ipynb` to Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Select **Runtime > Run all**.

The first execution downloads several gigabytes of model weights. An internet
connection and a GPU runtime are recommended. Generated results are written to:

```text
/content/multi_agent_rrg_demo/outputs/pipeline_results.json
```

## Data and reproducibility

The original research workflow used curated chest X-ray data and locally
prepared retrieval resources. Clinical datasets can require registration,
data-use agreements, institutional approval, controlled storage, and
redistribution restrictions.

To make the chapter example independently executable, the notebook instead
uses:

- one CC0 normal PA chest radiograph;
- one public-domain CDC pneumonia radiograph; and
- eight synthetic, de-identified teaching reports.

These two images are examples, not a representative evaluation dataset.
Notebook outputs are illustrative and cannot be used to estimate clinical
accuracy, fairness, robustness, or generalization. See
[`chapter-demo/sample_data/SOURCES.md`](chapter-demo/sample_data/SOURCES.md)
for image provenance.

## Model limitations

The open checkpoints were selected for accessibility and Colab compatibility,
not clinical performance. BiomedCLIP is not fine-tuned in this demonstration,
and the Qwen and SmolVLM checkpoints are general-purpose models. None of the
components or generated reports is clinically validated here.

## Repository layout

```text
.
├── agents/                 # Original five-agent implementation
├── models/                 # Original API wrapper
├── prompts/                # Original research prompts
├── run_train_pipeline.py   # Original pipeline entry point
└── chapter-demo/
    ├── multi_agent_rrg_demo.ipynb
    ├── README.md
    ├── prompts/            # Inspectable copies of all agent prompts
    ├── sample_data/        # Public examples, provenance, synthetic reports
    └── requirements.txt
```

## Scope and safety

This repository is intended for research and education. It is not a medical
device and must not be used for diagnosis, treatment, or clinical
decision-making.

For permanent chapter citation, archive a versioned release in the UNT Data
Repository or Zenodo and cite the resulting DOI rather than a moving branch.
