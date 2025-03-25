"""
Description: This module contains utility functions for loading and preparing data, 
training the quantum neural network model, and plotting the training history.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from torch.optim import Adam
import torch.nn as nn
import torch
import random
import os

from tqdm import tqdm

def set_seeds(seed=42):
    """
    Set seeds for reproducibility across all random number generators used within this framework.
    
    Args:
        seed (int): Seed value to use
    """
    # set all necessary seeds 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_and_prepare_data(X_path, y_path, bin_encoding=None, test_val_size=0.2, scaler=None, random_state=42, subset=None):
    """
    Load and prepare the data for training the quantum neural network.

    Args:
        X_path (str): Path to the input features
        y_path (str): Path to the target labels
        bin_encoding (dict): Binary encoding for the target labels
        test_val_size (float): Size of the test and validation sets
        scaler (object): Scaler object for normalizing the input features
        random_state (int): Random seed for reproducibility
        subset (int): Number of samples to use from the dataset

    Returns:
        X_train (torch.Tensor): Training features
        X_val (torch.Tensor): Validation features
        X_test (torch.Tensor): Test features
        y_train (torch.Tensor): Training targets
        y_val (torch.Tensor): Validation targets
        y_test (torch.Tensor): Test targets
        scaler (object): Scaler object used for normalizing the input features
    """
    # load the data
    X = np.load(X_path)
    y = np.load(y_path)
    
    # seperate the train set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, random_state=random_state, stratify=y
    )
    
    # get test and val from the temp set
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # apply scaling if provided
    if scaler is not None:            
        # Reshape if needed (for image data), fit on training, transform all sets
        orig_shape_train = X_train.shape
        orig_shape_val = X_val.shape
        orig_shape_test = X_test.shape
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_train = scaler.fit_transform(X_train_reshaped).reshape(orig_shape_train)
        
        X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(orig_shape_val)
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(orig_shape_test)

    # apply binary encoding if provided
    if bin_encoding is not None:
        y_train = np.array([bin_encoding[val] for val in y_train])
        y_val = np.array([bin_encoding[val] for val in y_val])
        y_test = np.array([bin_encoding[val] for val in y_test])


    if subset is not None:
        X_train = X_train[:subset]
        y_train = y_train[:subset]
        X_val = X_val[:subset]
        y_val = y_val[:subset]
        X_test = X_test[:subset]
        y_test = y_test[:subset]

    # convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler



def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=10, lr=0.01):
    """
    Train the quantum neural network model using PyTorch.
    
    Args:
        model (nn.Module): The quantum neural network model
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training targets
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
    
    Returns:
        model (nn.Module): Trained model
        losses (list): Training loss history
        val_losses (list): Validation loss history
        val_f1s (list): Validation F1 score history
    """
    # define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    # training loop
    losses = []
    val_losses = []
    val_f1s = []
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        # shuffle training data
        indices = torch.randperm(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        pbar = tqdm(total=n_batches, desc="Training progress", unit="batch")
        for batch in range(n_batches):
            # get mini-batch
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            y_pred, _ = model(X_batch)
            
            # compute loss
            loss = loss_fn(y_pred, y_batch)
            epoch_loss += loss.item()
            
            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

        # Record average loss
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)


        if X_val is not None and y_val is not None:                
            bce, accuracy, precision, recall, f1 = model.evaluate(X_val, y_val)
            val_losses.append(bce)
            val_f1s.append(f1)
            print(f"Epoch {epoch+1}/{epochs}, Train-Loss: {avg_loss:.4f}, Val-Loss: {bce:.4f}, Val-F1: {f1:.4f}, Val-Precision: {precision:.4f}, Val-Recall: {recall:.4f}, Val-Accuracy: {accuracy:.4f}")
        else: 
            print(f"Epoch {epoch+1}/{epochs}, Train-Loss: {avg_loss:.4f}")
    
    return model, losses, val_losses, val_f1s



def safe_json_load(df, run_id, column_name):
    """
    Helper function to safely load JSON data from a DataFrame.
    """
    try:
        return json.loads(df.loc[run_id, column_name])
    except Exception as e:
        print(f"Error loading {column_name}:", e)
        return None

def plot_training_history(df, run_id, save_path=None, figsize=(10, 5)):
    """
    Plot the training history.
    
    Args:
        df (pd.DataFrame): DataFrame containing the training history
        run_id (int): ID of the run
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
    """   
    # data prep

    loss_history = safe_json_load(df, run_id, "loss_history")
    val_loss_history = safe_json_load(df, run_id, "val_loss_history")
    val_f1s = safe_json_load(df, run_id, "val_f1s")

    # plotting
    if loss_history is not None and val_loss_history is not None:
        plt.figure(figsize=figsize)
        plt.plot(loss_history, label="Training Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross Entropy Loss")
        plt.legend()
        plt.title("Training History")
        if save_path is not None: # only save if save_path is provided
            loss_save_path = save_path + f"losses_ID{run_id}.png"
            plt.savefig(loss_save_path)
        plt.show()
    
    if val_f1s is not None:
        plt.figure(figsize=figsize)
        plt.plot(val_f1s, label="Validation F1 Score", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.title("Validation F1 Score")
        if save_path is not None: # only save if save_path is provided
            val_f1s_path = save_path + f"val_f1s_ID{run_id}.png"
            plt.savefig(val_f1s_path) 
        plt.show()

