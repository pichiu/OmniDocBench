# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OmniDocBench** is a comprehensive benchmark for evaluating document parsing in real-world scenarios. The system evaluates 1355 PDF pages across 9 document types, 4 layout types, and 3 languages, with rich annotations for blocks (20k+) and spans (80k+).

**Key Technologies:**
- Python 3.x with pandas, numpy, opencv-python
- Evaluation metrics: Edit Distance, TEDS, CDM, BLEU, METEOR, COCO mAP/mAR
- Configuration-driven YAML architecture
- Registry pattern for component management

## Language Policy

**Rationale**: This project serves a Taiwan-based user community while maintaining international collaboration capabilities through English technical documentation.

### Communication Language
Use **Traditional Chinese (Taiwan) / Mandarin (ZH-TW)** for all conversations and interactions with AI assistants.

### Documentation Language
- **English**:
  - `CLAUDE.md` (this file)
  - `GIT_WORKFLOW.md` (development workflow)
  - Commit messages (following Conventional Commits format)

- **Traditional Chinese (Taiwan) / ZH-TW**:
  - `/docs/` directory (project documentation)
  - New documentation files
  - API documentation and docstrings

- **Simplified Chinese (ZH-CN)** (existing):
  - `README_zh-CN.md`

- **Preserve Original**:
  - Third-party library documentation

### Code Implementation
- **Variable/function/class names**: English (snake_case/PascalCase)
- **Inline comments**: Traditional Chinese (Taiwan) preferred, English acceptable
- **Docstrings**: Traditional Chinese (Taiwan) for public APIs
- **Error messages**: English (for international compatibility)
- **Log messages**: English (for debugging and searchability)

## Common Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit and hooks
pip install pre-commit
pre-commit install

# Set up git commit template (optional)
git config commit.template .gitmessage
```

### Running Evaluations
```bash
# End-to-end evaluation
python task/end2end_run_eval.py --config configs/end2end.yaml

# Layout detection
python task/detection_eval.py --config configs/layout_detection.yaml

# Formula recognition
python task/recognition_eval.py --config configs/formula_recognition.yaml

# Table recognition
python task/recognition_eval.py --config configs/table_recognition.yaml

# Text OCR
python task/recognition_eval.py --config configs/ocr.yaml
```

### Viewing Results
```bash
# Results are saved in result/ directory
ls result/

# View JSON results
cat result/end2end_quick_match_metric_result.json | jq .
```

## Architecture

### Registry Pattern
The codebase uses a registry pattern for dynamic component management:

**Core registries** (`registry/registry.py`):
- `EVAL_TASK_REGISTRY` - Evaluation tasks
- `METRIC_REGISTRY` - Evaluation metrics
- `DATASET_REGISTRY` - Dataset loaders

**Usage example:**
```python
from registry.registry import EVAL_TASK_REGISTRY

@EVAL_TASK_REGISTRY.register("end2end_eval")
class End2EndEval:
    # Implementation
```

### Module Structure

```
configs/          # YAML configuration files for evaluation tasks
  ├── end2end.yaml
  ├── layout_detection.yaml
  ├── formula_recognition.yaml
  └── table_recognition.yaml

task/             # Evaluation task implementations
  ├── end2end_run_eval.py      # End-to-end evaluation
  ├── detection_eval.py         # Layout/formula detection
  └── recognition_eval.py       # OCR/formula/table recognition

dataset/          # Dataset loaders with matching algorithms
  ├── end2end_dataset.py
  ├── detection_dataset.py
  ├── recog_dataset.py
  └── md2md_dataset.py

metrics/          # Metric implementations
  ├── cal_metric.py             # Edit Distance, BLEU, METEOR
  ├── table_metric.py           # TEDS
  ├── cdm_metric.py             # CDM for formulas
  └── cdm/                      # CDM module

utils/            # Utilities and matching algorithms
  ├── match.py                  # Base matching logic
  ├── match_quick.py            # Quick matching algorithm
  ├── match_full.py             # Full matching algorithm
  ├── extract.py                # Content extraction
  └── ocr_utils.py              # OCR utilities

tools/            # Inference scripts for various models
  └── model_infer/              # Model-specific inference scripts
```

### Configuration-Driven Design

Evaluations are defined via YAML configs that specify:
- Task type (end2end_eval, layout_detection, etc.)
- Metrics to compute for each element type
- Dataset paths and matching methods
- Output naming

**Example config structure:**
```yaml
end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
    table:
      metric: [TEDS, Edit_dist]
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./demo_data/omnidocbench_demo/OmniDocBench_demo.json
    prediction:
      data_path: ./demo_data/end2end
    match_method: quick_match
```

## Adding New Components

### Adding a New Metric
1. Create metric class in `metrics/`
2. Register with `@METRIC_REGISTRY.register("metric_name")`
3. Implement `evaluate()` method returning (samples, results_dict)
4. Reference in config YAML

### Adding a New Task
1. Create task class in `task/`
2. Register with `@EVAL_TASK_REGISTRY.register("task_name")`
3. Implement dataset loading and metric coordination
4. Create corresponding YAML config

### Adding a New Dataset Loader
1. Create dataset class in `dataset/`
2. Register with `@DATASET_REGISTRY.register("dataset_name")`
3. Implement data loading and matching logic
4. Specify in config YAML

## Key Evaluation Metrics

- **Edit Distance**: Normalized Levenshtein distance for text comparison
- **TEDS**: Tree Edit Distance Similarity for table structure evaluation
- **CDM**: Character Detection Matching - visual matching metric for formulas (renders LaTeX to images and compares character bboxes)
- **BLEU/METEOR**: NLP metrics for text quality
- **COCO mAP/mAR**: Object detection metrics for layout detection

## Data Flow

1. **Config Loading** → YAML parsed to define task, metrics, dataset
2. **Dataset Loading** → Ground truth + predictions loaded via registered dataset class
3. **Matching** → GT and predictions matched (quick_match or full_match algorithms)
4. **Metric Computation** → Each registered metric processes matched samples
5. **Results Output** → JSON files saved to `result/` directory

## Important Notes

- The matching algorithm supports hybrid matching where formulas and text can match each other
- CDM metric requires special rendering environment for LaTeX → image conversion
- Results include overall scores plus breakdowns by document attributes and page types
- Pre-commit hooks run Black (formatting) and isort (import sorting)
