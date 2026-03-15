import argparse
import os
import pandas as pd
import numpy as np

from feature_model.feature_model import Feature_model
from feature_model.feature_model_functions import split_dataset_descriptor_both

def main():
    parser = argparse.ArgumentParser(description="DeSIDE-DDI Feature Generation Model")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help='Mode: train or predict')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV data')
    parser.add_argument('--model_type', type=int, choices=[1, 2, 3], default=3, help='1: Struct Only, 2: Prop Only, 3: Both')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_path', type=str, default='./trained_model/', help='Path to save/load model')
    parser.add_argument('--model_name', type=str, default='feature_model_weights', help='Model name prefix')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    input_data = pd.read_csv(args.data)
    
    x_fp_total, x_desc_total, y_total = split_dataset_descriptor_both(input_data)
    
    struct_only = (args.model_type == 1)
    prop_only = (args.model_type == 2)
    
    # Initialize the model
    model_obj = Feature_model(struct_only=struct_only, property_only=prop_only)
    
    if args.mode == 'train':
        print("Training model...")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            
        if args.model_type == 1:
            _x = x_fp_total.values
        elif args.model_type == 2:
            _x = x_desc_total.values
        else:
            _x = [x_fp_total.values, x_desc_total.values]
            
        model_obj.train(train_x=_x, train_y=y_total.values, 
                        model_save_path=args.save_path, 
                        model_name=args.model_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size)
        print("Training complete. Model saved.")
        
    elif args.mode == 'predict':
        print("Predicting with model...")
        model_obj.load_model(args.save_path, args.model_name + '.h5')
        
        if args.model_type == 1:
            _x = [x_fp_total.values]
        elif args.model_type == 2:
            _x = [x_desc_total.values]
        else:
            _x = [x_fp_total.values, x_desc_total.values]
            
        recon_x = model_obj.predict(_x)
        
        if len(y_total.columns) > 0:
            corr_list = []
            for i in range(recon_x.shape[0]):
                corr_list.append(np.corrcoef(y_total.values[i], recon_x[i])[0][1])
            print('Average Correlation with ground truth: ', np.mean(corr_list))
        else:
            print("Finished prediction. (No ground truth labels provided for correlation)")

if __name__ == '__main__':
    main()
