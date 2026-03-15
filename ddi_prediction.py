import argparse
import sys
import pandas as pd
import numpy as np

from ddi_model.data_load import *
from ddi_model.model import DDI_model
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="DeSIDE-DDI Prediction Model")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], required=True, help='Mode: train, test, or predict')
    parser.add_argument('--train_x', type=str, default='ddi_example_x.csv', help='Train inputs CSV name')
    parser.add_argument('--train_y', type=str, default='ddi_example_y.csv', help='Train labels CSV name')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory containing the data files')
    parser.add_argument('--save_path', type=str, default='./trained_model/', help='Path to save/load model')
    parser.add_argument('--model_name', type=str, default='ddi_model_weights', help='Model name prefix')
    parser.add_argument('--threshold_name', type=str, default='ddi_model_threshold.csv', help='Threshold file name for loading')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training per CV sample')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--sampling_size', type=int, default=1, help='Number of cross-validation samplings to perform')

    args = parser.parse_args()
    
    # 1. Initialize Model
    tf.random.set_seed(3)
    np.random.seed(3)
    ddi_model = DDI_model()
    
    # Load expression data
    try:
        ts_exp = load_exp(file_path=args.data_dir)
    except FileNotFoundError:
        print(f"Error: Could not find twosides_predicted_expression_scaled.csv in {args.data_dir}")
        sys.exit(1)

    if args.mode == 'train':
        print(f"Loading training data from {args.data_dir}...")
        train_data = load_train_example(file_path=args.data_dir, train_x_name=args.train_x, train_y_name=args.train_y)
        
        print("Starting training...")
        ddi_model.train(train_data=train_data, exp_df=ts_exp, split_frac=0.1, 
                        sampling_size=args.sampling_size, model_save_path=args.save_path, 
                        model_name=args.model_name, batch_size=args.batch_size)
                        
        ddi_model.optimal_threshold.to_csv(args.save_path + args.model_name + '_threshold.csv')
        print(f"Training complete. Weights and thresholds saved to {args.save_path}.")

    elif args.mode == 'test':
        print(f"Loading test data from {args.data_dir}...")
        train_data = load_train_example(file_path=args.data_dir, train_x_name=args.train_x, train_y_name=args.train_y)
        
        print(f"Loading pre-trained model '{args.model_name}'...")
        ddi_model.load_model(model_load_path=args.save_path, model_name=args.model_name + '.h5', threshold_name=args.threshold_name)
        
        print("Evaluating model performance on provided test dataset...")
        predicted_label, performance = ddi_model.test(
            test_x=train_data.iloc[:,:3], test_y=train_data.iloc[:,3], 
            exp_df=ts_exp, batch_size=args.batch_size
        )
        print("Performance Results:")
        print(performance[['SN', 'SP', 'PR', 'AUC', 'AUPR']].astype(float).describe())
        
        predicted_label.to_csv(args.save_path + args.model_name + '_test_predictions.csv')
        print(f"Predictions saved to {args.save_path}{args.model_name}_test_predictions.csv")

if __name__ == '__main__':
    main()
