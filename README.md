# Multi-Agent Radiology Report Generation

This repository contains the original five-agent radiology report-generation
pipeline and a self-contained educational notebook developed as executable
companion code for a book chapter.

[Open the chapter demo in Google Colab](https://colab.research.google.com/drive/1Oru5c9Ah_vtLe0TN22VtPep3sUUjGlOt?usp=sharing)

## Original research implementation

The repository root preserves the initial research implementation for historical
and methodological reference. Its five stages used different infrastructure
from the chapter notebook:

- the Top-K retrieval agent used OpenCLIP together with a locally trained
  retrieval checkpoint and dataset-specific JSON files;
- the General, Critical, and Summarizing agents called an OpenAI-hosted model;
- the Image Agent called a locally installed LLaVA-Med checkpoint; and
- the orchestration script contained local filesystem paths for prompts,
  images, checkpoints, and outputs.

That code documents the original system design, but it is not a turnkey public
demo because it depends on API credentials, locally prepared weights, and
controlled or separately obtained data. The self-contained notebook retains
the same conceptual agent roles while replacing those dependencies with open
models, synthetic retrieval text, and public teaching images.

## Chapter companion

The recommended entry point is
[`chapter-demo/multi_agent_rrg_demo.ipynb`](chapter-demo/multi_agent_rrg_demo.ipynb).
It is a single-file notebook that can run from top to bottom on a Google Colab
T4 GPU without an API key or access to a controlled clinical dataset.

The notebook:

1. installs its dependencies;
2. sets deterministic seeds;
3. downloads openly licensed teaching radiographs;
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

## Disclaimer

This repository and its generated outputs are provided solely for research,
education, and software demonstration. They are not medical advice, are not a
medical device, and have not been validated for diagnosis, treatment, triage,
or any other clinical decision. The models can omit findings, invent findings,
misinterpret images, or produce inconsistent reports.

Do not submit protected health information or identifiable patient data to this
demonstration. Any use with clinical or controlled data requires appropriate
authorization, governance, security controls, expert oversight, and independent
validation. The authors and contributors make no warranty regarding accuracy,
fitness for a particular purpose, safety, or clinical performance.

## License

New material under `chapter-demo/` is released under the
[MIT License](chapter-demo/LICENSE), except for third-party images, model
weights, software dependencies, and other resources that retain their own
licenses and terms. Image-specific attribution and licensing are documented in
[`chapter-demo/sample_data/SOURCES.md`](chapter-demo/sample_data/SOURCES.md).

The historical research code outside `chapter-demo/` is preserved from the
original project and is **not relicensed** by the chapter-demo MIT License.
Users are responsible for confirming that they have permission for their
intended use of that historical code and all third-party components.

## Contact

Questions, reproducibility reports, and corrections are welcome:

**Jinyu Liu** — [jinyuliu@my.unt.edu](mailto:jinyuliu@my.unt.edu)

For permanent chapter citation, archive a versioned release in the UNT Data
Repository or Zenodo and cite the resulting DOI rather than a moving branch.
