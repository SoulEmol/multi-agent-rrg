# Reproducible five-agent radiology report generation demo

This directory is the executable companion for the chapter. It adapts the original five-agent research pipeline into a self-contained educational demonstration that requires no API key and no controlled dataset.

## Run in Google Colab

1. Upload only `multi_agent_rrg_demo.ipynb` to Colab, or open it from GitHub.
2. In Colab, select **Runtime > Change runtime type > T4 GPU**.
3. Open `multi_agent_rrg_demo.ipynb` and choose **Runtime > Run all**.

The first run downloads approximately 4 GB of openly available model weights. A GPU runtime and internet connection are required. No source-code edits, secrets, or manual data downloads are required.

## Pipeline

1. Retrieval Agent: BiomedCLIP cross-modal retrieval.
2. Draft Agent: retrieval-grounded preliminary report.
3. Refiner Agent: evidence audit.
4. Vision Agent: direct image description with SmolVLM.
5. Synthesis Agent: evidence-aware final report.

The single-file notebook embeds its prompts and synthetic retrieval collection, downloads two openly licensed radiographs at runtime, and writes structured results to `/content/multi_agent_rrg_demo/outputs/pipeline_results.json` in Colab.

## Scope and safety

This software is an educational demonstration, not a medical device. Its outputs are not validated for diagnosis, treatment, or clinical decision-making. The compact open models were selected for reproducibility and accessibility, not clinical performance.

The companion `sample_data/SOURCES.md`, `prompts/`, and `requirements.txt` files are retained for inspection and archival clarity, but the notebook does not require them at runtime.
