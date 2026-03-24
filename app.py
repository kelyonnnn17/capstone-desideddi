import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from graph.builder import build_reasoning_graph
import networkx as nx

from rag.data_loader import load_rag_data
from rag.retriever import retrieve_context

import tensorflow as tf
from ddi_model.model import DDI_model
from ddi_model.data_load import load_exp

# Prevent TF from eating all GPU memory when UI launches
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
CORS(app)

print("Starting Server Boot Process...")

# 1. LOAD DATA GLOBALS 
DATA_DIR = './data/'
MODEL_DIR = './trained_model/'
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/chat')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'cniongolo/biomistral')
DEFAULT_DEMOGRAPHIC_CONTEXT = os.environ.get(
    'DESIDE_DEMOGRAPHIC_CONTEXT',
    'observational polypharmacy cohorts with skew toward older adults, patients with multimorbidity, and prescribing patterns shaped by comorbidity burden'
)
OLLAMA_SYSTEM_PROMPT = (
    "You are an expert clinical pharmacologist and computational biologist. "
    "Assume the two drugs are being co-administered concurrently. "
    "Do not ask clarifying questions. "
    "Follow the requested Step 1, Step 2, Step 3 structure exactly. "
    "Be decisive, distinguish mechanism from observational bias, and end with "
    "\"Verified Biological Effect\" and \"Confidence Assessment\"."
)
OLLAMA_REPAIR_PROMPT = (
    "Rewrite the prior answer into the exact required format. "
    "Do not ask any questions. Do not restate the prompt. "
    "Output only these sections: Step 1, Step 2, Step 3, Verified Biological Effect, Confidence Assessment."
)
OLLAMA_TEMPLATE_REPAIR_PROMPT = (
    "Fill every section of this template with complete content. "
    "Use 2-4 sentences per step. Do not output only headings.\n"
    "Step 1: <complete answer>\n"
    "Step 2: <complete answer>\n"
    "Step 3: <complete answer>\n"
    "Verified Biological Effect: <complete answer>\n"
    "Confidence Assessment: <complete answer>"
)
LOG_LLM_PIPELINE = os.environ.get('DESIDE_LOG_LLM_PIPELINE', '1').lower() in ('1', 'true', 'yes', 'on')
LOG_LLM_FULL_PAYLOAD = os.environ.get('DESIDE_LOG_LLM_FULL_PAYLOAD', '0').lower() in ('1', 'true', 'yes', 'on')
LOG_LLM_MAX_CHARS = int(os.environ.get('DESIDE_LOG_LLM_MAX_CHARS', '12000'))
NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NCBI_USER_AGENT = "DeSIDE-DDI/1.0 (local grounding for pharmacology review)"
OLLAMA_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "step_1": {"type": "string"},
        "step_2": {"type": "string"},
        "step_3": {"type": "string"},
        "verified_biological_effect": {"type": "string"},
        "confidence_assessment": {"type": "string"}
    },
    "required": [
        "step_1",
        "step_2",
        "step_3",
        "verified_biological_effect",
        "confidence_assessment"
    ]
}

# A dictionary caching exactly what we need for predictions
state = {
    'ddi_model': None,
    'ts_exp': None,
    'drugs': [],
    'drug_lookup': {},
    'se_names': [],
    'thresholds': None,
    'gene_annotation': None,
    'train_x_dummy': None,
    'train_y_dummy': None
}

def _truncate_text(value, max_chars=2000):
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"

def _log_json_block(title, payload, max_chars=None):
    if not LOG_LLM_PIPELINE:
        return
    limit = max_chars or LOG_LLM_MAX_CHARS
    try:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        rendered = f"<failed to serialize log payload: {exc}>"
    print(f"\n===== {title} =====\n{_truncate_text(rendered, limit)}\n===== END {title} =====\n")

def _build_llm_payload_log_preview(llm_payload):
    top_results = llm_payload.get("predictions", {}).get("top_results", [])
    top_genes = llm_payload.get("model_features", {}).get("top_input_genes", [])
    top_se_embeddings = llm_payload.get("model_features", {}).get("top_side_effect_embeddings", [])
    preview = {
        "input": llm_payload.get("input", {}),
        "predictions": {
            "total_predicted_reactions": llm_payload.get("predictions", {}).get("total_predicted_reactions", 0),
            "top_results": top_results[:10]
        },
        "model_features": {
            "feature_source": llm_payload.get("model_features", {}).get("feature_source"),
            "drug1_expression_vector_len": len(llm_payload.get("model_features", {}).get("drug1_expression_vector", [])),
            "drug2_expression_vector_len": len(llm_payload.get("model_features", {}).get("drug2_expression_vector", [])),
            "drug1_shared_embedding_len": len(llm_payload.get("model_features", {}).get("drug1_shared_embedding", [])),
            "drug2_shared_embedding_len": len(llm_payload.get("model_features", {}).get("drug2_shared_embedding", [])),
            "drug1_pair_embedding_len": len(llm_payload.get("model_features", {}).get("drug1_pair_embedding", [])),
            "drug2_pair_embedding_len": len(llm_payload.get("model_features", {}).get("drug2_pair_embedding", [])),
            "top_input_genes_preview": top_genes[:15],
            "top_side_effect_embeddings_preview": [
                {
                    "se_id": item.get("se_id"),
                    "name": item.get("name"),
                    "confidence_gap": item.get("confidence_gap"),
                    "probability_score": item.get("probability_score"),
                    "se_embedding_len": len(item.get("se_embedding", [])),
                    "se_head_embedding_len": len(item.get("se_head_embedding", [])),
                    "se_tail_embedding_len": len(item.get("se_tail_embedding", []))
                }
                for item in top_se_embeddings[:10]
            ]
        },
        "llm_prompt": llm_payload.get("llm_prompt", "")
    }
    if LOG_LLM_FULL_PAYLOAD:
        preview["full_llm_payload"] = llm_payload
    return preview

def boot_ai_engine():
    print("-> Loading RAG Knowledge Base...")
    load_rag_data()

    print("-> Loading Expression Matrix...")
    try:
        state['ts_exp'] = load_exp(file_path=DATA_DIR)
    except FileNotFoundError:
        print("ERROR: Missing twosides_predicted_expression_scaled.csv in data block!")

    print("-> Pre-loading DDI Tensorflow 2 Model...")
    tf.random.set_seed(3)
    np.random.seed(3)
    
    state['ddi_model'] = DDI_model()
    # Explicitly load weights from trained checkpoint
    try:
        state['ddi_model'].load_model(model_load_path=MODEL_DIR, model_name='ddi_model_weights.h5', threshold_name='ddi_model_threshold.csv')
        state['thresholds'] = state['ddi_model'].optimal_threshold
    except Exception as e:
        print(f"ERROR LOAD MODEL FILES: {e}")

    print("-> Loading Drug and Side Effect Data Mappings...")
    try:
        # Load dictionary mappings for users to select by text name rather than ID
        drugs_df = pd.read_csv(os.path.join(DATA_DIR, 'twosides_drug_info.csv'))
        # Using PubchemID and Name format
        state['drugs'] = drugs_df[['pubchemID', 'name']].to_dict(orient='records')
        state['drug_lookup'] = dict(zip(drugs_df['pubchemID'].astype(int), drugs_df['name']))

        se_df = pd.read_csv(os.path.join(DATA_DIR, 'twosides_side_effect_info.csv'))
        # SE mapping connects integer IDs (0 to 962) from the Threshold dataframe to real names
        state['se_names'] = se_df.set_index('SE_map')['Side Effect Name'].to_dict()
        
        # Load Gene mappings for LLM Context Extraction
        state['gene_annotation'] = pd.read_csv(os.path.join(DATA_DIR, 'lincs_gene_list.csv'), index_col=0)
        state['train_x_dummy'] = pd.read_csv(os.path.join(DATA_DIR, 'ddi_example_x.csv'))
        state['train_y_dummy'] = pd.read_csv(os.path.join(DATA_DIR, 'ddi_example_y.csv'))
        
    except Exception as e:
        print(f"ERROR LOAD MAPPINGS: {e}")
        
    print("Boot Process Complete. Backend AI Online.")

# Initialize the state before requests come in
boot_ai_engine()

# --- ROUTES ---

def _get_drug_name(drug_id):
    return state['drug_lookup'].get(int(drug_id), f"Drug {drug_id}")

def _lookup_expression_vector(drug_id):
    exp_row = state['ts_exp'][state['ts_exp']['pubchem'] == int(drug_id)]
    if exp_row.empty:
        raise ValueError(f"Expression vector not found for Drug {drug_id}.")
    return exp_row.drop(columns=['pubchem']).iloc[0].astype(float)

def _build_test_frame(d1_id, d2_id):
    test_x = pd.DataFrame({
        'drug1': [int(d1_id)] * 963,
        'drug2': [int(d2_id)] * 963,
        'SE': list(range(963))
    })
    test_y = pd.Series([0] * 963, name='label')
    return test_x, test_y

def _run_prediction(d1_id, d2_id):
    test_x, test_y = _build_test_frame(d1_id, d2_id)
    predicted_label_df, _ = state['ddi_model'].test(
        test_x=test_x,
        test_y=test_y,
        exp_df=state['ts_exp'],
        batch_size=1024
    )
    positives = predicted_label_df[predicted_label_df['final_predicted_label'] == 1].sort_values(by='gap', ascending=False)
    return predicted_label_df, positives

def _extract_model_features(d1_id, d2_id, positives):
    model = state['ddi_model'].model
    shared_layer = model.get_layer('drug_embed_shared')
    shared_layer2 = model.get_layer('drug_embed_shared2')
    glu1_layer = model.get_layer('drug1_glu')
    glu2_layer = model.get_layer('drug2_glu')
    se_emb_layer = model.get_layer('se_emb')
    se_head_layer = model.get_layer('se_head')
    se_tail_layer = model.get_layer('se_tail')

    d1_series = _lookup_expression_vector(d1_id)
    d2_series = _lookup_expression_vector(d2_id)
    d1_vec = d1_series.to_numpy(dtype=np.float32).reshape(1, -1)
    d2_vec = d2_series.to_numpy(dtype=np.float32).reshape(1, -1)

    d1_shared = shared_layer(d1_vec, training=False).numpy()
    d2_shared = shared_layer(d2_vec, training=False).numpy()
    concat_vec = np.concatenate([d1_shared, d2_shared], axis=1).astype(np.float32)

    d1_gate = glu1_layer(concat_vec, training=False).numpy()
    d2_gate = glu2_layer(concat_vec, training=False).numpy()

    bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    if len(bn_layers) < 2:
        raise ValueError("Expected pair-conditioning batch normalization layers were not found.")

    d1_selected = bn_layers[0](d1_shared * d1_gate, training=False).numpy()
    d2_selected = bn_layers[1](d2_shared * d2_gate, training=False).numpy()

    d1_pair_embedding = shared_layer2(d1_selected, training=False).numpy().flatten().tolist()
    d2_pair_embedding = shared_layer2(d2_selected, training=False).numpy().flatten().tolist()

    gene_annotation = state['gene_annotation']
    gene_names = gene_annotation.iloc[:, 0].astype(str).tolist()
    combined_abs = np.abs(d1_series.values) + np.abs(d2_series.values)
    top_gene_indices = np.argsort(combined_abs)[::-1][:50]
    top_input_genes = [
        {
            "gene_index": int(idx),
            "gene_name": gene_names[idx] if idx < len(gene_names) else f"Gene {idx}",
            "drug1_expression": round(float(d1_series.iloc[idx]), 6),
            "drug2_expression": round(float(d2_series.iloc[idx]), 6),
            "combined_abs_expression": round(float(combined_abs[idx]), 6)
        }
        for idx in top_gene_indices
    ]

    top_side_effect_embeddings = []
    for _, row in positives.head(10).iterrows():
        se_idx = int(row['SE'])
        se_tensor = np.array([[se_idx]], dtype=np.int32)
        top_side_effect_embeddings.append({
            "se_id": se_idx,
            "name": state['se_names'].get(se_idx, f"Unknown SE #{se_idx}").capitalize(),
            "confidence_gap": round(float(row['gap']), 6),
            "probability_score": round(float(row['mean_predicted_score']), 6),
            "se_embedding": se_emb_layer(se_tensor).numpy().reshape(-1).tolist(),
            "se_head_embedding": se_head_layer(se_tensor).numpy().reshape(-1).tolist(),
            "se_tail_embedding": se_tail_layer(se_tensor).numpy().reshape(-1).tolist()
        })

    return {
        "feature_source": "twosides_predicted_expression_scaled.csv",
        "drug1_expression_vector": d1_series.round(6).tolist(),
        "drug2_expression_vector": d2_series.round(6).tolist(),
        "drug1_shared_embedding": np.round(d1_shared.flatten(), 6).tolist(),
        "drug2_shared_embedding": np.round(d2_shared.flatten(), 6).tolist(),
        "drug1_pair_embedding": [round(float(v), 6) for v in d1_pair_embedding],
        "drug2_pair_embedding": [round(float(v), 6) for v in d2_pair_embedding],
        "top_input_genes": top_input_genes,
        "top_side_effect_embeddings": top_side_effect_embeddings
    }

def _build_llm_payload(d1_id, d2_id, positives):
    top_results = []
    for _, row in positives.head(30).iterrows():
        se_idx = int(row['SE'])
        top_results.append({
            'se_id': se_idx,
            'name': state['se_names'].get(se_idx, f"Unknown SE #{se_idx}").capitalize(),
            'confidence_gap': round(float(row['gap']), 4),
            'probability_score': round(float(row['mean_predicted_score']), 4)
        })

    model_features = _extract_model_features(d1_id, d2_id, positives)
    summary_lines = [
        f"Analyze the interaction between {_get_drug_name(d1_id)} (CID {d1_id}) and {_get_drug_name(d2_id)} (CID {d2_id}).",
        "Use the predicted side effects and the exported model vectors to explain likely pathways and safety risks."
    ]
    if top_results:
        summary_lines.append(
            "Top predicted adverse reactions: " +
            "; ".join(f"{item['name']} (gap {item['confidence_gap']}, score {item['probability_score']})" for item in top_results[:10])
        )

    return {
        "input": {
            "drug1": {"id": int(d1_id), "name": _get_drug_name(d1_id)},
            "drug2": {"id": int(d2_id), "name": _get_drug_name(d2_id)}
        },
        "predictions": {
            "total_predicted_reactions": int(len(positives)),
            "top_results": top_results
        },
        "model_features": model_features,
        "llm_prompt": " ".join(summary_lines)
    }

def _build_analysis_prompt(llm_payload, predicted_side_effect=None, demographic_context=None):
    predictions = llm_payload["predictions"]["top_results"]
    selected_effect = predicted_side_effect or (predictions[0]["name"] if predictions else "no high-confidence side effect predicted")
    top_genes = llm_payload["model_features"]["top_input_genes"][:15]
    genes_text = ", ".join(
        f"{item['gene_name']} (drug1={item['drug1_expression']}, drug2={item['drug2_expression']})"
        for item in top_genes
    ) or "No genes were available."

    drug_a = llm_payload["input"]["drug1"]["name"]
    drug_b = llm_payload["input"]["drug2"]["name"]
    context = demographic_context or DEFAULT_DEMOGRAPHIC_CONTEXT

    return f"""Evaluate this as a concurrent co-administration scenario and provide a direct answer without asking clarifying questions.

Step 1: Independent Baseline Analysis
Based exclusively on established biomedical literature and pharmacokinetic principles, what is the known interaction between {drug_a} and {drug_b}? Describe the primary molecular pathways involved and the most biologically plausible adverse effects of combining these two drugs.

Step 2: Evaluate Predictive Model Hypothesis
A predictive algorithm trained on observational data (which contains demographic skews, such as {context}) has hypothesized that combining {drug_a} and {drug_b} causes the following side effect: {selected_effect}.

To justify this prediction, the model's attention mechanism flagged the perturbation of the following genes: {genes_text}.

Step 3: Adjudication and Final Synthesis
Cross-reference your independent analysis from Step 1 with the algorithm's hypothesis in Step 2.
1. Do the specific genes flagged by the model biologically map to {selected_effect}?
2. Is {selected_effect} a genuine pharmacological consequence of this drug combination, or is it highly likely to be a statistical artifact driven by the demographic data (for example, the condition is naturally prevalent in the demographic taking these drugs)?

Provide a final, definitive explanation that separates the true molecular interaction from observational noise.

Required output format:
Step 1: ...
Step 2: ...
Step 3: ...
Verified Biological Effect: ...
Confidence Assessment: ..."""

def _fetch_json(url):
    request_obj = urllib.request.Request(url, headers={"User-Agent": NCBI_USER_AGENT})
    with urllib.request.urlopen(request_obj, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))

def _fetch_text(url):
    request_obj = urllib.request.Request(url, headers={"User-Agent": NCBI_USER_AGENT})
    with urllib.request.urlopen(request_obj, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")

def _get_pubmed_evidence(drug_a, drug_b, max_articles=3):
    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "retmode": "json",
        "retmax": max_articles,
        "sort": "relevance",
        "term": f'("{drug_a}"[Title/Abstract]) AND ("{drug_b}"[Title/Abstract]) AND (interaction OR bleeding OR pharmacokinetic OR pharmacodynamic)'
    })
    search_url = f"{NCBI_EUTILS_BASE}esearch.fcgi?{search_params}"
    try:
        search_data = _fetch_json(search_url)
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        summary_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        })
        summary_url = f"{NCBI_EUTILS_BASE}esummary.fcgi?{summary_params}"
        summary_data = _fetch_json(summary_url).get("result", {})

        fetch_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text"
        })
        fetch_url = f"{NCBI_EUTILS_BASE}efetch.fcgi?{fetch_params}"
        abstract_text = _fetch_text(fetch_url)
        abstract_chunks = [chunk.strip() for chunk in abstract_text.split("\n\n") if chunk.strip()]

        evidence = []
        for idx, pmid in enumerate(pmids):
            summary = summary_data.get(pmid, {})
            title = summary.get("title", f"PMID {pmid}")
            snippet = abstract_chunks[idx][:250] if idx < len(abstract_chunks) else ""
            evidence.append({
                "pmid": pmid,
                "title": title,
                "snippet": snippet
            })
        return evidence
    except Exception as exc:
        print(f"PubMed grounding unavailable for {drug_a} + {drug_b}: {exc}")
        return []

def _safe_trim(text, max_chars=300):
    return text[:max_chars] if text else ""

def _call_ollama_messages(messages, temperature=0.1, num_predict=140):
    request_body = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive" : 0,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": num_predict
        }
    }).encode("utf-8")

    request_obj = urllib.request.Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request_obj, timeout=180) as response:
        parsed = json.loads(response.read().decode("utf-8"))

    message = parsed.get("message") or {}
    return {
        "model": parsed.get("model", OLLAMA_MODEL),
        "response": (message.get("content") or parsed.get("response") or "").strip(),
        "done": parsed.get("done", True),
        "total_duration": parsed.get("total_duration")
    }

def _looks_like_prompt_echo(text):
    if not text:
        return True
    lowered = text.lower()
    forbidden_fragments = [
        "based exclusively on established biomedical literature",
        "a predictive algorithm trained on observational data",
        "cross-reference your independent analysis",
        "what is the known interaction between",
        "do the specific genes flagged",
        "provide a final, definitive explanation",
        "the user's response is satisfactory",
        "\"step_1\"",
        "\"step_2\"",
        "\"step_3\""
    ]
    return len(text.strip()) < 60 or any(fragment in lowered for fragment in forbidden_fragments)

def _generate_text_section(user_prompt, retry_prompt, min_length=80, num_predict=260, section_name="section"):
    _log_json_block(
        f"LLM COT INPUT - {section_name}",
        {
            "system_prompt": OLLAMA_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "min_length": min_length,
            "num_predict": num_predict
        },
        max_chars=LOG_LLM_MAX_CHARS
    )

    messages = [
        {"role": "system", "content": OLLAMA_SYSTEM_PROMPT + " Return only the requested answer content, not JSON or headings."},
        {"role": "user", "content": user_prompt}
    ]
    result = _call_ollama_messages(messages, temperature=0.1, num_predict=num_predict)
    text = result["response"]
    repaired = False

    if len(text.strip()) < min_length or _looks_like_prompt_echo(text):
        repaired = True
        _log_json_block(
            f"LLM COT RETRY TRIGGER - {section_name}",
            {
                "reason": "short_output_or_prompt_echo",
                "first_response": text,
                "retry_prompt": retry_prompt
            },
            max_chars=LOG_LLM_MAX_CHARS
        )
        retry_messages = messages + [
            {"role": "assistant", "content": text or "No valid answer was produced."},
            {"role": "user", "content": retry_prompt}
        ]
        retry_result = _call_ollama_messages(retry_messages, temperature=0.05, num_predict=num_predict)
        text = retry_result["response"]
        result = retry_result

    final_section = {
        "text": text.strip(),
        "model": result["model"],
        "done": result["done"],
        "total_duration": result["total_duration"],
        "repaired": repaired
    }
    _log_json_block(
        f"LLM COT OUTPUT - {section_name}",
        final_section,
        max_chars=LOG_LLM_MAX_CHARS
    )
    return final_section

def _generate_chain_of_thought_analysis(llm_payload, predicted_side_effect=None, demographic_context=None):
    _log_json_block(
        "DDI OUTPUTS PASSED TO LLM",
        _build_llm_payload_log_preview(llm_payload),
        max_chars=LOG_LLM_MAX_CHARS
    )

    predictions = llm_payload["predictions"]["top_results"]
    selected_effect = predicted_side_effect or (predictions[0]["name"] if predictions else "no high-confidence side effect predicted")
    top_genes = llm_payload["model_features"]["top_input_genes"][:12]
    genes_text = ", ".join(
        f"{item['gene_name']} (drug1={item['drug1_expression']}, drug2={item['drug2_expression']})"
        for item in top_genes
    ) or "No genes were available."

    drug_a = llm_payload["input"]["drug1"]["name"]
    drug_b = llm_payload["input"]["drug2"]["name"]


    rag_query = f"{drug_a} {drug_b} drug interaction mechanism"
    rag_docs = retrieve_context(rag_query)
    rag_text = "\n".join(rag_docs)

    context = demographic_context or DEFAULT_DEMOGRAPHIC_CONTEXT
    pubmed_evidence = _get_pubmed_evidence(drug_a, drug_b)
    evidence_text = "\n".join(
        f"- PMID {item['pmid']}: {item['title']}. {item['snippet']}"
        for item in pubmed_evidence
    ) if pubmed_evidence else "No pair-specific PubMed evidence was retrieved; rely on conservative general pharmacology knowledge and state uncertainty when needed."

    step1 = _generate_text_section(
        user_prompt=(
            f"Concurrent co-administration case.\nDrug A: {drug_a}\nDrug B: {drug_b}\n"
            f"Retrieved biomedical knowledge:\n{rag_text}\n"
            f"Pair-specific literature evidence:\n{evidence_text}\n"
            "Write Step 1 only as a direct paragraph of 2 to 4 sentences. "
            "Use the literature evidence first, and only then general pharmacology. "
            "Explain the known or most plausible pharmacokinetic and pharmacodynamic interaction, "
            "the main molecular pathways, and the most biologically plausible adverse effects. "
            "Do not ask questions. Do not repeat the instructions."
        ),
        retry_prompt=(
            f"Your previous Step 1 answer was invalid because it copied instructions or was incomplete. "
            f"Answer directly about {drug_a} plus {drug_b} in 2 to 4 sentences, grounded in this evidence:\n{evidence_text}"
        ),
        min_length=120,
        num_predict=320,
        section_name="Step 1"
    )

    step2 = _generate_text_section(
        user_prompt=(
            f"Concurrent co-administration case.\nDrug A: {drug_a}\nDrug B: {drug_b}\n"
            f"Predicted side effect: {selected_effect}\nFlagged genes: {genes_text}\n"
            "Write Step 2 only as a direct paragraph of 2 to 4 sentences. "
            "Evaluate whether these genes plausibly support the predicted side effect and what biological processes they suggest. "
            "If support is weak or nonspecific, say so clearly. Do not repeat the instructions."
        ),
        retry_prompt=(
            f"Your previous Step 2 answer was invalid because it copied instructions or was incomplete. "
            f"Directly evaluate whether the flagged genes plausibly support {selected_effect} for {drug_a} plus {drug_b}."
        ),
        min_length=120,
        num_predict=260,
        section_name="Step 2"
    )

    step3 = _generate_text_section(
        user_prompt=(
            f"Concurrent co-administration case.\nDrug A: {drug_a}\nDrug B: {drug_b}\n"
            f"Demographic context: {context}\n"
            f"Retrieved biomedical knowledge: {rag_text}\n"
            f"Pair-specific literature evidence: {evidence_text}\n"
            f"Baseline analysis: {_safe_trim(step1['text'])}\n"
            f"Gene-based evaluation: {_safe_trim(step2['text'])}\n"
            f"Predicted side effect: {selected_effect}\n"
            "Write Step 3 only as a direct paragraph of 3 to 5 sentences. "
            "Adjudicate whether the predicted side effect is a genuine pharmacological consequence or likely observational noise. "
            "Be decisive and separate true mechanism from demographic artifact."
        ),
        retry_prompt=(
            f"Your previous Step 3 answer was invalid because it copied instructions or was incomplete. "
            f"Give a decisive adjudication for whether {selected_effect} is real pharmacology or likely observational noise for {drug_a} plus {drug_b}."
        ),
        min_length=160,
        num_predict=320,
        section_name="Step 3"
    )

    verdict = _generate_text_section(
        user_prompt=(
            f"Drug pair: {drug_a} + {drug_b}\n"
            f"Predicted side effect under review: {selected_effect}\n"
            f"Step 1 draft: {step1['text']}\n"
            f"Step 2 draft: {step2['text']}\n"
            f"Step 3 draft: {step3['text']}\n"
            "Return exactly two lines and nothing else:\n"
            "Verified Biological Effect: <one concise sentence>\n"
            "Confidence Assessment: <one concise sentence>"
        ),
        retry_prompt=(
            "Your previous verdict was invalid. Return exactly these two lines only:\n"
            "Verified Biological Effect: <one concise sentence>\n"
            "Confidence Assessment: <one concise sentence>"
        ),
        min_length=60,
        num_predict=140,
        section_name="Verdict"
    )

    verdict_lines = [line.strip() for line in verdict["text"].splitlines() if line.strip()]
    verified_line = next((line for line in verdict_lines if line.startswith("Verified Biological Effect:")), None)
    confidence_line = next((line for line in verdict_lines if line.startswith("Confidence Assessment:")), None)

    if not verified_line:
        verified_line = "Verified Biological Effect: The strongest supported interaction should be taken from the pharmacology in Step 1 rather than the model label alone."
    if not confidence_line:
        confidence_line = "Confidence Assessment: Moderate confidence in the mechanistic summary; low confidence in the model-predicted side effect unless Step 2 and Step 3 support it."

    analysis_text = "\n\n".join([
        f"Step 1: {step1['text']}",
        f"Step 2: {step2['text']}",
        f"Step 3: {step3['text']}",
        verified_line,
        confidence_line
    ])

    simple_summary = _generate_text_section(
        user_prompt=(
            f"Drug pair: {drug_a} + {drug_b}\n"
            f"Predicted side effect under review: {selected_effect}\n"
            f"Baseline mechanism: {step1['text']}\n"
            f"Gene-based evaluation: {step2['text']}\n"
            f"Final adjudication: {step3['text']}\n"
            f"Verified line: {verified_line}\n"
            f"Confidence line: {confidence_line}\n"
            "Write one short final answer for a user. "
            "Use 2 to 3 sentences only. Strip away demographic bias and observational noise. "
            "State the biologically supported interaction, and clearly say whether the predicted side effect is truly supported or likely an artifact. "
            "Do not use headings, bullets, JSON, or quote the prompt."
        ),
        retry_prompt=(
            f"Your previous summary was invalid because it copied instructions or was incomplete. "
            f"Give one short bias-stripped conclusion for {drug_a} plus {drug_b} in 2 to 3 sentences only."
        ),
        min_length=80,
        num_predict=180,
        section_name="Simple Summary"
    )

    final_prediction = _generate_text_section(
        user_prompt=(
            f"Drug pair: {drug_a} + {drug_b}\n"
            f"Predicted side effect under review: {selected_effect}\n"
            f"Baseline mechanism: {step1['text']}\n"
            f"Gene-based evaluation: {step2['text']}\n"
            f"Final adjudication: {step3['text']}\n"
            f"Verified line: {verified_line}\n"
            f"Confidence line: {confidence_line}\n"
            "Return exactly one line and nothing else in this format:\n"
            "Final Prediction: <state the most biologically supported real interaction, and whether the model-predicted side effect is supported or likely artifact>"
        ),
        retry_prompt=(
            "Your previous final prediction was invalid. Return exactly one line in this format:\n"
            "Final Prediction: <one concise sentence>"
        ),
        min_length=40,
        num_predict=120,
        section_name="Final Prediction"
    )

    final_prediction_line = next(
        (line.strip() for line in final_prediction["text"].splitlines() if line.strip().startswith("Final Prediction:")),
        None
    )
    if not final_prediction_line:
        final_prediction_line = (
            f"Final Prediction: The best-supported clinical concern for {drug_a} plus {drug_b} should be taken from the "
            "mechanistic interaction in Step 1, while the model-predicted side effect should be treated cautiously unless Steps 2 and 3 clearly support it."
        )

    total_duration = sum(item["total_duration"] or 0 for item in [step1, step2, step3, verdict, simple_summary])
    total_duration += final_prediction["total_duration"] or 0
    repaired = any(item["repaired"] for item in [step1, step2, step3, verdict, simple_summary, final_prediction])

    return {
        "response": f"{analysis_text}\n\n{final_prediction_line}",
        "simple_response": simple_summary["text"],
        "model": step1["model"] or OLLAMA_MODEL,
        "done": all(item["done"] for item in [step1, step2, step3, verdict, simple_summary, final_prediction]),
        "total_duration": total_duration,
        "repaired": repaired
    }

def _build_prediction_summary(d1_id, d2_id, llm_payload):
    return {
        "drug1_requested": int(d1_id),
        "drug2_requested": int(d2_id),
        "total_predicted_reactions": llm_payload["predictions"]["total_predicted_reactions"],
        "top_results": llm_payload["predictions"]["top_results"]
    }

@app.route('/')
def home():
    """Serves the main frontend index.html from templates format"""
    return render_template('index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Returns the dropdown arrays of Drug choices"""
    return jsonify({
        "status": "success",
        "drugs": state['drugs']
    })

@app.route('/api/predict', methods=['POST'])
def predict_ddi():
    """
    Receives Drug1 and Drug2 pubchem IDs, predicts interaction severities 
    for all 963 SE IDs, and returns a sorted list of the highest probability side effects.
    """
    data = request.json
    d1_id = data.get('drug1')
    d2_id = data.get('drug2')
    
    if not d1_id or not d2_id:
         return jsonify({"error": "Must provide 'drug1' and 'drug2' PubChem IDs via POST."}), 400

    print(f"Request: Predicting Side Effects for [Drug {d1_id} + Drug {d2_id}]")
    
    if state['ts_exp'] is None or state['ddi_model'] is None:
        return jsonify({"error": "AI Backend improperly initialized."}), 500
        
    try:
        _, positives = _run_prediction(d1_id, d2_id)
        llm_payload = _build_llm_payload(d1_id, d2_id, positives)

        return jsonify({
            "status": "success",
            "drug1_requested": d1_id,
            "drug2_requested": d2_id,
            "total_predicted_reactions": len(positives),
            "top_results": llm_payload["predictions"]["top_results"],
            "llm_payload": llm_payload
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_llm', methods=['POST'])
def export_llm():
    """
    Returns a structured JSON payload containing predictions and model vectors
    so the UI can pass the full inference context to an LLM.
    """
    data = request.json
    d1_id = data.get('drug1')
    d2_id = data.get('drug2')
    
    if not d1_id or not d2_id:
         return jsonify({"error": "Must provide 'drug1' and 'drug2'"}), 400

    print(f"LLM Context Request for [Drug {d1_id} + Drug {d2_id}]")
    
    try:
        _, positives = _run_prediction(d1_id, d2_id)
        return jsonify({
            "status": "success",
            "llm_payload": _build_llm_payload(d1_id, d2_id, positives)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze_llm', methods=['POST'])
def analyze_llm():
    """
    Builds the pharmacology adjudication prompt and sends it to the local Ollama model.
    """
    data = request.json or {}
    d1_id = data.get('drug1')
    d2_id = data.get('drug2')
    predicted_side_effect = data.get('predicted_side_effect')
    demographic_context = data.get('demographic_context')
    include_payload = bool(data.get('include_payload', False))
    include_prompt = bool(data.get('include_prompt', False))

    if not d1_id or not d2_id:
        return jsonify({"error": "Must provide 'drug1' and 'drug2'."}), 400

    print(f"LLM Analysis Request for [Drug {d1_id} + Drug {d2_id}] via {OLLAMA_MODEL}")

    try:
        _, positives = _run_prediction(d1_id, d2_id)
        llm_payload = _build_llm_payload(d1_id, d2_id, positives)
        prompt = _build_analysis_prompt(
            llm_payload=llm_payload,
            predicted_side_effect=predicted_side_effect,
            demographic_context=demographic_context
        )
        ollama_result = _generate_chain_of_thought_analysis(
            llm_payload=llm_payload,
            predicted_side_effect=predicted_side_effect,
            demographic_context=demographic_context
        )
        response_payload = {
            "status": "success",
            **_build_prediction_summary(d1_id, d2_id, llm_payload),
            "analysis": ollama_result["response"],
            "simple_analysis": ollama_result.get("simple_response", ollama_result["response"]),
            "selected_side_effect": predicted_side_effect or (llm_payload["predictions"]["top_results"][0]["name"] if llm_payload["predictions"]["top_results"] else None),
            "ollama": {
                "url": OLLAMA_URL,
                "model": ollama_result["model"],
                "done": ollama_result["done"],
                "total_duration": ollama_result["total_duration"],
                "repaired": bool(ollama_result.get("repaired", False))
            }
        }
        if include_payload:
            response_payload["llm_payload"] = llm_payload
        if include_prompt:
            response_payload["analysis_prompt"] = prompt
        return jsonify(response_payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/graph', methods=['POST'])
def get_graph():
    data = request.json
    d1 = data['drug1']
    d2 = data['drug2']

    def extract_id(drug):
        if isinstance(drug, dict):
            return (
                drug.get('id') or
                drug.get('pubchemID') or
                drug.get('value')
            )
        return drug

    d1_id = extract_id(d1)
    d2_id = extract_id(d2)

    if d1_id is None or d2_id is None:
        print("GRAPH BAD INPUT:", data)
        return jsonify({"error": "Invalid drug input"}), 400

    _, positives = _run_prediction(d1_id, d2_id)
    llm_payload = _build_llm_payload(d1_id, d2_id, positives)

    drug_a = _get_drug_name(d1_id)
    drug_b = _get_drug_name(d2_id)

    rag_docs = retrieve_context(f"{drug_a} {drug_b} interaction")

    G = build_reasoning_graph(
        drug_a,
        drug_b,
        llm_payload["predictions"]["top_results"],
        llm_payload["model_features"]["top_input_genes"],
        rag_docs
    )

    return jsonify(nx.node_link_data(G))


if __name__ == '__main__':
    # Launch on Port 5003, visible to all local networks
    app.run(host='0.0.0.0', port=5003, debug=True, use_reloader=False)
