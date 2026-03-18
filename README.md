# DeSIDE-DDI

DeSIDE-DDI is a TensorFlow 2 based pipeline for drug-drug interaction (DDI) side-effect prediction from drug-induced expression signatures, plus an optional LLM adjudication layer for mechanistic interpretation.

This repository includes:
- A pre-trained DDI model (963 side-effect classes).
- A pre-trained feature-generation model (predicts 978-gene expression from molecular descriptors/fingerprints).
- A Flask web app with interactive UI.
- CLI scripts for training, testing, exporting payloads, and analysis.

## What The System Does

At a high level, the project runs in two stages.

1. Feature model (optional in this repo runtime)
- Input: chemical fingerprints and descriptors.
- Output: predicted 978-gene expression signature per drug.

2. DDI model (used by the web app)
- Input: drug A expression, drug B expression, and side-effect ID.
- Output: interaction score for that side effect.
- Decision: compare score to side-effect-specific threshold.

In the web app, the backend evaluates all 963 side effects for a selected drug pair, ranks positives, and can send a structured context to a local Ollama model for narrative adjudication.

## Repository Layout

```text
DeSIDE-DDI/
	app.py                          Flask backend and API endpoints
	data/                           Input datasets and mappings
	ddi_model/                      DDI neural network + data loader + evaluation helpers
	feature_model/                  Feature-generation neural network + helpers
	scripts/                        CLI tools (train/test/export/analyze)
	static/                         Frontend JS/CSS
	templates/                      Frontend HTML
	trained_model/                  Pretrained weights and thresholds
	analysis_output/                Saved analysis CSV/plots
	run_mac_linux.sh                One-command launcher for macOS/Linux
	run_windows.bat                 One-command launcher for Windows
	requirements.txt                Python dependencies
```

## Data And Shapes

Observed from the bundled data files:
- `data/twosides_predicted_expression_scaled.csv`: 645 drugs, each with 978 expression features + `pubchem` key.
- `data/twosides_drug_info.csv`: 645 drug metadata rows with `pubchemID` and name.
- `data/twosides_side_effect_info.csv`: 963 side effects with `SE_map` index.
- `data/ddi_example_x.csv`: columns `drug1,drug2,SE`.
- `data/ddi_example_y.csv`: column `label`.
- `data/example_feature_model_input.csv`: feature-model input table (fingerprints + 100 descriptors + labels).

## Model Internals

### DDI model (`ddi_model/model.py`)

- Drug expression input dim: 978.
- Side-effect embedding table size: 963.
- Learns shared drug embedding, gated pair-conditioning, and side-effect-specific head/tail mappings.
- Predicts a distance-like score per `(drug1, drug2, SE)` triple.
- Uses per-side-effect optimal thresholds loaded from `trained_model/ddi_model_threshold.csv`.

Prediction behavior:
- Evaluates original pair `(drug1, drug2, SE)` and switched pair `(drug2, drug1, SE)`.
- Averages the two predicted scores.
- Applies side-effect threshold to get final label.

### Feature model (`feature_model/feature_model.py`)

- Supports three modes:
	- `model_type=1`: structure only (1024-d fingerprints).
	- `model_type=2`: property only (100 descriptors).
	- `model_type=3`: both (default).
- Predicts 978-d expression vector with MSE + Pearson metric.

## Web App Runtime Flow

When `app.py` starts:
1. Loads expression matrix (`load_exp`).
2. Loads DDI model weights and thresholds.
3. Loads drug and side-effect mapping tables.
4. Loads gene annotations used for LLM feature context.

When user submits a pair:
1. Backend builds 963 test rows (`SE=0..962`).
2. Runs DDI model inference and keeps positive side effects.
3. Builds top ranked response payload.
4. Optionally extracts intermediate embeddings and top genes for LLM context.
5. If calling analysis endpoint, queries PubMed (NCBI E-utilities) and sends structured prompts to Ollama.

## API Endpoints

Base URL default: `http://127.0.0.1:5003`

### `GET /api/metadata`
Returns the list of drugs for dropdowns.

### `POST /api/predict`
Body:
```json
{ "drug1": 2244, "drug2": 3672 }
```
Returns ranked predictions and LLM payload scaffold.

### `POST /api/export_llm`
Body:
```json
{ "drug1": 2244, "drug2": 3672 }
```
Returns the full structured payload (predictions + vectors + embeddings).

### `POST /api/analyze_llm`
Body example:
```json
{
	"drug1": 2244,
	"drug2": 3672,
	"predicted_side_effect": "bleeding",
	"demographic_context": "older adults with multimorbidity",
	"include_payload": false,
	"include_prompt": false
}
```
Runs staged adjudication via Ollama and returns:
- `analysis` (full Step1/Step2/Step3 output)
- `simple_analysis`
- selected side effect
- model metadata

## Setup

## 1) Quick launch scripts

macOS/Linux:
```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

Windows:
```bat
run_windows.bat
```

Both launchers create/activate `venv`, install requirements (first run), and start Flask on port 5003.

## 2) Manual setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

On Apple Silicon, TensorFlow Metal may be needed for native GPU acceleration:
```bash
pip install tensorflow-macos tensorflow-metal
```

## Ollama Integration (Optional But Required For UI Analyze Button)

The app defaults to:
- `OLLAMA_URL=http://127.0.0.1:11434/api/chat`
- `OLLAMA_MODEL=cniongolo/biomistral`

Example local Ollama setup:
```bash
ollama pull cniongolo/biomistral
ollama serve
```

Environment variables:
- `OLLAMA_URL`: override Ollama chat endpoint.
- `OLLAMA_MODEL`: choose a different local model.
- `DESIDE_DEMOGRAPHIC_CONTEXT`: default context string injected into adjudication prompts.
- `DESIDE_LOG_LLM_PIPELINE`: `1/0` to enable prompt/response logging.
- `DESIDE_LOG_LLM_FULL_PAYLOAD`: `1/0` to log full payloads.
- `DESIDE_LOG_LLM_MAX_CHARS`: truncate log size.

Note:
- The main UI button currently calls `/api/analyze_llm`, so Ollama availability affects end-to-end UI prediction flow.
- If you only want numeric predictions, call `/api/predict` directly.

## CLI Usage

Activate environment first:
```bash
source venv/bin/activate
```

### Feature model

Train:
```bash
python scripts/feature_generation.py \
	--mode train \
	--data data/example_feature_model_input.csv \
	--model_type 3 \
	--epochs 10 \
	--batch_size 64 \
	--save_path ./trained_model/ \
	--model_name feature_model_weights
```

Predict/reconstruction:
```bash
python scripts/feature_generation.py \
	--mode predict \
	--data data/example_feature_model_input.csv \
	--model_type 3 \
	--save_path ./trained_model/ \
	--model_name feature_model_weights
```

### DDI model

Train:
```bash
python scripts/ddi_prediction.py \
	--mode train \
	--data_dir ./data/ \
	--train_x ddi_example_x.csv \
	--train_y ddi_example_y.csv \
	--save_path ./trained_model/ \
	--model_name ddi_model_weights \
	--sampling_size 1 \
	--batch_size 1024
```

Test:
```bash
python scripts/ddi_prediction.py \
	--mode test \
	--data_dir ./data/ \
	--train_x ddi_example_x.csv \
	--train_y ddi_example_y.csv \
	--save_path ./trained_model/ \
	--model_name ddi_model_weights \
	--threshold_name ddi_model_threshold.csv
```

### Analysis script

```bash
python scripts/feature_analysis.py \
	--drug_id 3310 \
	--data_dir ./data/ \
	--model_dir ./trained_model/ \
	--output_dir ./analysis_output/
```

Outputs include top genes CSV and clustermap image.

### Payload/export helper scripts

Export payload JSON from running backend:
```bash
python scripts/export_llm_payload.py 2244 3672 --url http://127.0.0.1:5003/api/export_llm
```

Run adjudication call and save JSON:
```bash
python scripts/analyze_with_ollama.py 2244 3672 --url http://127.0.0.1:5003/api/analyze_llm
```

## Included Artifacts

Bundled in `trained_model/`:
- `ddi_model_weights.h5`
- `ddi_model_threshold.csv`
- `feature_model_weights.h5`
- `ddi_model_weights_test_predictions.csv`

Typical generated outputs:
- `analysis_output/drug_<id>_top100_genes.csv`
- `analysis_output/drug_<id>_clustermap.png`
- `analysis_<drug1>_<drug2>.json`
- `llm_payload_<drug1>_<drug2>.json`

## Troubleshooting

- Server starts but UI fails on prediction:
	- Check Ollama server/model availability if using the default UI flow.
- `FileNotFoundError` for expression matrix:
	- Ensure `data/twosides_predicted_expression_scaled.csv` exists.
- TensorFlow GPU memory spikes:
	- `app.py` enables memory growth on detected GPUs; confirm proper GPU backend install.
- macOS script fails on `python3.11`:
	- Install Python 3.11 or edit launcher to use your installed Python executable.

## Known Limitations

- `scripts/ddi_prediction.py` parser lists `predict` mode, but only `train` and `test` branches are implemented.
- The web app uses model outputs plus optional LLM narrative; LLM text is not a clinical decision support system.

## Citation And License

Original methodology by Eunyoung Kim.

This repository includes:
- `LICENSE`
- `CC-BY-NC-SA-4.0`

Use and distribution should comply with those license files.
