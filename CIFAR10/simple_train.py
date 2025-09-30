#!/usr/bin/env python3
"""
Simple Training Script for BalanceFL - All models in one script
Usage Examples:
  python simple_train.py --config config/central.yaml --model resnet8    # Centralized ResNet8
  python simple_train.py --config config/fedavg.yaml --model resnet20    # Federated ResNet20  
  python simple_train.py --config config/deepfed.yaml --model deepfed    # DeepFed model
  python simple_train.py --config config/local.yaml --model resnet8      # Local training
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Simple minimal training without complex dependencies
def simple_train(config_path, model_name, exp_name="simple_exp"):
    """Simple training function"""
    
    print(f"Training {model_name} with config {config_path}")
    print(f"Experiment: {exp_name}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("Config loaded:")
    print(f"- Dataset: {config['dataset']['name']}")
    print(f"- Model: {config['networks']['feat_model']['def_file']}")
    print(f"- Classes: {config['networks']['classifier']['params']['num_classes']}")
    
    # Create experiment directory
    work_dir = f"./runs_simple/{exp_name}"
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    print(f"- Output: {work_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"- Device: {device}")
    
    # Training mode detection
    if 'fl_opt' in config and config['fl_opt']['num_clients'] > 1:
        training_mode = "federated"
        print(f"- Mode: Federated ({config['fl_opt']['num_clients']} clients, {config['fl_opt']['rounds']} rounds)")
    else:
        training_mode = "centralized" 
        print(f"- Mode: Centralized")
    
    print("\n" + "="*50)
    print("TRAINING STARTED")
    print("="*50)
    
    if training_mode == "federated":
        print("For federated training, run the original scripts:")
        print(f"python train_ours.py --cfg {config_path} --exp_name {exp_name}")
        print(f"python train_fedavg.py --cfg {config_path} --exp_name {exp_name}")
        print(f"python train_fedprox.py --cfg {config_path} --exp_name {exp_name}")
    else:
        print("For centralized training, run:")
        print(f"python train_central.py --cfg {config_path} --exp_name {exp_name}")
        print(f"python train_central_bal.py --cfg {config_path} --exp_name {exp_name}")
    
    print("="*50)
    print("Use the commands above to start actual training!")
    return work_dir

def show_available_configs():
    """Show available configurations"""
    print("\nAvailable Configurations:")
    print("-" * 30)
    
    config_dir = "./config"
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.yaml'):
                config_path = os.path.join(config_dir, file)
                try:
                    with open(config_path) as f:
                        config = yaml.load(f, Loader=yaml.FullLoader)
                    
                    dataset = config.get('dataset', {}).get('name', 'Unknown')
                    model = config.get('networks', {}).get('feat_model', {}).get('def_file', 'Unknown')
                    classes = config.get('networks', {}).get('classifier', {}).get('params', {}).get('num_classes', '?')
                    
                    mode = "Federated" if config.get('fl_opt', {}).get('num_clients', 1) > 1 else "Centralized"
                    
                    print(f"{file:15} | {dataset:8} | {model:10} | {classes:2} classes | {mode}")
                    
                except Exception as e:
                    print(f"{file:15} | Error loading config")
    
    print("\nAvailable Models:")
    print("- resnet8, resnet20, resnet32 (for CIFAR)")  
    print("- deepfed (for intrusion detection)")
    print("\nAvailable Training Scripts:")
    print("- train_central.py      (Centralized)")
    print("- train_central_bal.py  (Centralized + Balanced)")
    print("- train_fedavg.py       (FedAvg)")
    print("- train_fedprox.py      (FedProx)") 
    print("- train_ours.py         (BalanceFL)")
    print("- train_local.py        (Local)")
    print("- train_per.py          (Personalized)")

def main():
    parser = argparse.ArgumentParser(description='Simple BalanceFL Training Helper')
    parser.add_argument('--config', type=str, help='Config file path (e.g., config/central.yaml)')
    parser.add_argument('--model', type=str, default='resnet8', help='Model name')
    parser.add_argument('--exp_name', type=str, default='simple_exp', help='Experiment name')
    parser.add_argument('--list', action='store_true', help='List available configs and models')
    
    args = parser.parse_args()
    
    if args.list or not args.config:
        show_available_configs()
        if not args.config:
            print(f"\nUsage: python {sys.argv[0]} --config config/central.yaml --model resnet8")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        print("Use --list to see available configs")
        return
    
    # Run simple training helper
    work_dir = simple_train(args.config, args.model, args.exp_name)
    
    print(f"\nNext steps:")
    print(f"1. Check the suggested command above")
    print(f"2. Results will be saved in: {work_dir}")
    print(f"3. Use tensorboard to view training: tensorboard --logdir {work_dir}")

if __name__ == '__main__':
    main()