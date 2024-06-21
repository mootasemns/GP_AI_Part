import os
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from utils.model import CheatDetectionModel
from utils import readConfig
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm for progress bar


def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, device):
    """Train the model

    Args:
        model (nn.Module): The neural network model to train
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer for the model
        num_epochs (int): Number of epochs to train
        dataloaders (dict): Dataloaders for training and validation
        dataset_sizes (dict): Sizes of the training and validation datasets
        device (torch.device): Device to run the training on (CPU or GPU)

    Returns:
        nn.Module: Trained model
    """
    best_acc = 0.0  # Variable to store the best accuracy achieved during training
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_precisions = []
    val_precisions = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            running_loss = 0.0  # Variable to accumulate loss during the epoch
            running_corrects = 0  # Variable to accumulate number of correct predictions
            predictions = []
            ground_truth = []
            
            # Use tqdm for progress visualization
            data_loader = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}/{num_epochs - 1}')
            
            for inputs, labels in data_loader:
                inputs = inputs.to(device)  # Move inputs to the appropriate device
                labels = labels.to(device)  # Move labels to the appropriate device
                
                optimizer.zero_grad()  # Zero the parameter gradients
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Get model outputs
                    _, preds = torch.max(outputs, 1)  # Get the predicted classes
                    loss = criterion(outputs, labels)  # Calculate loss
                    
                    # Backward pass and optimization (only in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    if phase == 'val':
                        predictions.extend(preds.cpu().numpy())
                        ground_truth.extend(labels.cpu().numpy())
            
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update tqdm progress bar description
                data_loader.set_postfix(loss=running_loss / dataset_sizes[phase], 
                                        acc=running_corrects.double().item() / dataset_sizes[phase])  # Convert tensor to scalar
            
            epoch_loss = running_loss / dataset_sizes[phase]  # Calculate epoch loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # Calculate epoch accuracy
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                train_precisions.append(precision_score(ground_truth, predictions, average='macro'))
            elif phase == 'val':
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                val_precisions.append(precision_score(ground_truth, predictions, average='macro'))
                
                # Log metrics to MLflow
                mlflow.log_metric('val_loss', epoch_loss)
                mlflow.log_metric('val_acc', epoch_acc.item())
                mlflow.log_metric('val_precision', precision_score(ground_truth, predictions, average='macro'))
                
                # Save the model if it has the best accuracy so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), model_path)
    
    # Plotting
    epochs = range(num_epochs)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # Ensure the plots folder exists
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save the plots
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'))
    plt.show()
    return model  # Return the trained model

if __name__ == '__main__':
    # Load configurations
    train_config = readConfig('config/train_config.yaml')
    model_config = readConfig('config/model_config.yaml')
    model_path = model_config["MODEL"]["model_path"]
    
    # Define transforms for training and validation datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    
    # Load the dataset from 'train' and 'validate' folders inside 'processed_data'
    train_dir = os.path.join(train_config['train']['data_dir'], 'train')
    val_dir = os.path.join(train_config['train']['data_dir'], 'valid')
    
    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    
    # Create dataloaders for training and validation
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=train_config['train']['batch_size'], shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=train_config['train']['batch_size'], shuffle=False, num_workers=0)
    }
    
    # Get the sizes of the datasets
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    # Initialize model, loss function (criterion), and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CheatDetectionModel(model_config['MODEL']['num_classes'])  # Initialize custom model
    model = model.to(device)  # Move model to the appropriate device
    # criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss
    criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss 
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer
    
    # MLflow tracking
    mlflow.set_experiment('cheat_detection')  # Set the MLflow experiment
    with mlflow.start_run():
        mlflow.log_param('epochs', train_config['train']['epochs'])
        mlflow.log_param('batch_size', train_config['train']['batch_size'])
        
        # Train the model
        model = train_model(model, criterion, optimizer, train_config['train']['epochs'], dataloaders, dataset_sizes, device)
        
        # Log the best model to MLflow
        mlflow.pytorch.log_model(model, 'models')
