import argparse
import os
import sys
# Add parent directory to module search path so local imports resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from feature_model.feature_model import Feature_model
from ddi_model.model import DDI_model

def find_exp(drug_df, ts_exp, column_name):
    return pd.merge(drug_df, ts_exp, left_on=column_name, right_on='pubchem', how='left').iloc[:,2:]

def extract_expression(classification_model, train_data, drug_id, twosides_exp):
    att_a = train_data[(train_data.drug1 == drug_id)&(train_data.label == 1)].drop_duplicates(subset=['drug1','drug2'])
    att_b = train_data[(train_data.drug2 == drug_id)&(train_data.label == 1)].drop_duplicates(subset=['drug1','drug2'])
    
    att_b2 = att_b[['drug2','drug1','SE','label']]
    att_b2.columns = ['drug1','drug2','SE','label']
    
    att_drug_pair = pd.concat([att_a, att_b2], axis=0, ignore_index=True)
    
    drug1 = find_exp(att_drug_pair[['drug1']], twosides_exp, 'drug1')
    drug2 = find_exp(att_drug_pair[['drug2']], twosides_exp, 'drug2')
    drug_se = att_drug_pair['SE']
    
    se_list_gb_a = pd.DataFrame(pd.merge(att_a[['drug1','drug2']], train_data, on=['drug1','drug2'], how='left').groupby(['drug1','drug2'])['SE'].apply(list))
    se_list_gb_b = pd.DataFrame(pd.merge(att_b[['drug1','drug2']], train_data, on=['drug1','drug2'], how='left').groupby(['drug1','drug2'])['SE'].apply(list))
    
    se_list_gb = pd.concat([se_list_gb_a, se_list_gb_b], axis=0)
    se_binary = pd.DataFrame(np.zeros((se_list_gb.shape[0], 963)), columns=[x for x in range(0,963)], index=se_list_gb.index)
    for index, row in se_list_gb.iterrows():
        se_binary.loc[index, row.SE] = 1
    
    print('Data extraction ready, building Keras functions...')
    
    print('Data extraction ready, building Keras functions...')
    
    # In TF2, use Model(inputs, outputs) to extract intermediate layers by name
    # inputs = [drug1_exp, drug2_exp, input_se]
    in1 = classification_model.get_layer('drug1_exp').input
    in2 = classification_model.get_layer('drug2_exp').input
    in3 = classification_model.get_layer('input_se').input
    inputs = [in1, in2, in3]
    
    # Extract outputs from the explicitly named layers
    # layers[7] -> multiply_1 (drug2_selected), layers[4] -> drug_embed_shared(drug1), layers[8] -> drug_embed_shared2 (drug1_emb)
    # The names might vary slightly, but we can grab the functional layers that were named.
    # Since we need the intermediate tensors, we can create a sub-model that outputs all layers
    intermediate_model = keras.Model(inputs=classification_model.inputs, outputs=[l.output for l in classification_model.layers])
    
    # Predict all layer outputs
    all_outputs = intermediate_model.predict([drug1.values.astype(float), drug2.values.astype(float), drug_se.values])
    
    # Find indices of the specific layers we want from the original sequence 
    # [classification_model.layers[7].output, classification_model.layers[8].output, classification_model.layers[4].output]
    attention_vector = all_outputs[7]
    attention_vector2 = all_outputs[8]
    d1_vector = all_outputs[4]
    
    return att_drug_pair, attention_vector, attention_vector2, drug1, d1_vector, se_binary

def extract_top100genes(ddi_pair, gene_header):
    temp = pd.DataFrame(ddi_pair, columns=gene_header.values.flatten().tolist())
    top_100_gene = pd.DataFrame({col: temp.abs().T[col].nlargest(100).index.tolist() for n, col in enumerate(temp.T)}).T
    return top_100_gene

def main():
    parser = argparse.ArgumentParser(description="DeSIDE-DDI Feature Analysis Module")
    parser.add_argument('--drug_id', type=int, default=3310, help='PubChem Drug ID to analyze (e.g. 3310)')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='./trained_model/', help='Trained DDI model path')
    parser.add_argument('--output_dir', type=str, default='./analysis_output/', help='Directory to output plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Loading data...")
    try:
        gene_annotation = pd.read_csv(os.path.join(args.data_dir, 'lincs_gene_list.csv'), index_col=0)
        twosides_exp = pd.read_csv(os.path.join(args.data_dir, 'twosides_predicted_expression_scaled.csv'))
        train_x = pd.read_csv(os.path.join(args.data_dir, 'ddi_example_x.csv'))
        train_y = pd.read_csv(os.path.join(args.data_dir, 'ddi_example_y.csv'))
        train_data = pd.concat([train_x, train_y], axis=1)
    except Exception as e:
        print(f"Error loading required files from {args.data_dir}: {e}")
        return
        
    print("Loading pre-trained DDI model...")
    ddi_obj = DDI_model()
    try:
        ddi_obj.load_model(model_load_path=args.model_dir, model_name='ddi_model_weights.h5', threshold_name='ddi_model_threshold.csv')
    except Exception as e:
        print(f"Error loading model weights. Make sure ddi_model_weights.h5 is in {args.model_dir}. Error: {e}")
        return

    print(f"Extracting features for Drug ID {args.drug_id}...")
    try:
        drug_pair_subset, att_mat1, att_mat2, input_d1_mat, d1_att_mat, se_binary = extract_expression(ddi_obj.model, train_data, args.drug_id, twosides_exp)
    except Exception as e:
        print(f"Extraction failed (check if drug {args.drug_id} is in subset): {e}")
        return

    drug1_sig_genes = extract_top100genes(att_mat1, gene_annotation)
    genes_out = os.path.join(args.output_dir, f'drug_{args.drug_id}_top100_genes.csv')
    drug1_sig_genes.to_csv(genes_out)
    print(f"Significant genes saved to {genes_out}")
    
    print("Generating Heatmap...")
    plt.figure()
    temp = pd.DataFrame(d1_att_mat)
    temp = temp.fillna(0)
    cg = sns.clustermap(temp, metric='correlation', figsize=(18,10), row_cluster=True, col_cluster=True, 
                   cmap='seismic', vmin=-2, vmax=2, center=0, xticklabels=False, yticklabels=False)
    
    hm_ax = cg.ax_heatmap
    hm_ax.add_patch(Rectangle((0,0), 978, temp.shape[0], fill=False, edgecolor='grey', lw=3))
    cg.ax_col_dendrogram.set_visible(False)
    
    plot_out = os.path.join(args.output_dir, f'drug_{args.drug_id}_clustermap.png')
    plt.savefig(plot_out)
    print(f"Plot saved to {plot_out}")

if __name__ == '__main__':
    main()
