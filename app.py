import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

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

# A dictionary caching exactly what we need for predictions
state = {
    'ddi_model': None,
    'ts_exp': None,
    'drugs': [],
    'se_names': [],
    'thresholds': None
}

def boot_ai_engine():
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

        se_df = pd.read_csv(os.path.join(DATA_DIR, 'twosides_side_effect_info.csv'))
        # SE mapping connects integer IDs (0 to 962) from the Threshold dataframe to real names
        state['se_names'] = se_df.set_index('SE_map')['Side Effect Name'].to_dict()
    except Exception as e:
        print(f"ERROR LOAD MAPPINGS: {e}")
        
    print("Boot Process Complete. Backend AI Online.")

# Initialize the state before requests come in
boot_ai_engine()

# --- ROUTES ---

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
        # Re-create a mock test dataframe structure holding every SE instance
        # We test [d1, d2] for SE[0...962]
        test_x = pd.DataFrame({
            'drug1': [int(d1_id)] * 963,
            'drug2': [int(d2_id)] * 963,
            'SE': list(range(963))
        })
        # Dummy labels for testing pipeline execution (won't be evaluated)
        test_y = pd.Series([0]*963, name='label')

        # Run via the test wrapper
        # The internal functions use average predictions between (d1,d2) and (d2,d1)
        predicted_label_df, _ = state['ddi_model'].test(
            test_x=test_x, 
            test_y=test_y, 
            exp_df=state['ts_exp'], 
            batch_size=1024 # Custom batches
        )
        
        # We only want the ones where the AI flagged 'final_predicted_label' == 1
        positives = predicted_label_df[predicted_label_df['final_predicted_label'] == 1]
        
        # Sort by furthest gap away from the threshold (highest confidence of reacting)
        positives = positives.sort_values(by='gap', ascending=False)
        
        # Limit the results so we don't overwhelm the UI, grab top N 
        results = []
        for index, row in positives.head(30).iterrows():
            se_idx = int(row['SE'])
            ui_name = state['se_names'].get(se_idx, f"Unknown SE #{se_idx}")
            
            results.append({
                'se_id': se_idx,
                'name': ui_name.capitalize(),
                'confidence_gap': round(row['gap'], 4),
                'probability_score': round(row['mean_predicted_score'], 4)
            })
            
        return jsonify({
            "status": "success",
            "drug1_requested": d1_id,
            "drug2_requested": d2_id,
            "total_predicted_reactions": len(positives),
            "top_results": results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Launch on Port 5002, visible to all local networks
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)

