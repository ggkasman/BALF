import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
import random
from lr_one_cycle import OneCycleScheduler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Paths
_TRAIN_DIR = '../Data/Training/'
_VAL_DIR = '../Data/Validation/'
_DATA_SELECTOR = '*.png'
_SAMPLE_SIZE = None  # Use all training data
_BATCH_SIZE = 1  # Original batch size
_LEARNING_RATE = 1e-5
_LOG_DIR = '../Model/'

# Class weights for imbalanced data
_CLASS_WEIGHTS = torch.tensor([0.91, 5.90, 1.13, 0.54], device=device)

class WeightedL1Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    
    def forward(self, pred, target):
        # Apply weights to each class's loss
        loss = torch.abs(pred - target) * self.weights.unsqueeze(0)
        return loss.mean()

class CellDataset(Dataset):
    def __init__(self, data_dir, transform=None, cache_images=False):
        self.data_dir = pathlib.Path(data_dir)
        self.transform = transform
        self.cache_images = cache_images
        self.image_paths = list(self.data_dir.rglob(_DATA_SELECTOR))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
        
        # Initialize caches
        self.labels_cache = {}
        self.image_cache = {} if cache_images else None
        
        # Pre-load labels
        for img_path in self.image_paths:
            # Cache labels (small memory footprint)
            label_path = str(img_path).replace('.png', '_labels.txt')
            try:
                with open(label_path, 'r') as f:
                    self.labels_cache[str(img_path)] = torch.tensor(
                        [float(x) for x in f.read().strip().split(',')],
                        dtype=torch.float32
                    )
            except FileNotFoundError:
                # Try alternative label file name
                label_path = str(img_path.parent / 'labels.txt')
                with open(label_path, 'r') as f:
                    self.labels_cache[str(img_path)] = torch.tensor(
                        [float(x) for x in f.read().strip().split(',')],
                        dtype=torch.float32
                    )
            
            # Cache images if requested (for validation set)
            if self.cache_images:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.image_cache[str(img_path)] = image
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_path_str = str(img_path)
        
        # Get image (from cache or load from disk)
        if self.cache_images:
            image = self.image_cache[img_path_str]
        else:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        # Get cached labels
        labels = self.labels_cache[img_path_str]
        
        return image, labels

class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet50 with the new API
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Replace final layer
        self.resnet.fc = nn.Linear(2048, 4)
        self.softmax = nn.Softmax(dim=1)
        
        # Enable gradient computation for all parameters
        for param in self.resnet.parameters():
            param.requires_grad = True
            
    def freeze_base_model(self):
        # Freeze all layers except final layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_base_model(self):
        # Unfreeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.softmax(x)
        return x

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders():
    # Create datasets (cache validation images since they're few and don't use augmentation)
    train_dataset = CellDataset(_TRAIN_DIR, transform=get_transforms(is_train=True), cache_images=False)
    val_dataset = CellDataset(_VAL_DIR, transform=get_transforms(is_train=False), cache_images=True)
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=_BATCH_SIZE,
        shuffle=True,
        num_workers=1,  # Keep original worker setting
        pin_memory=True if torch.cuda.is_available() else False,  # Only use pin_memory with GPU
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False  # Only use pin_memory with GPU
    )
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        running_loss += loss.item()
        
        # Only print final loss for the epoch
        if i == total_batches - 1:
            print(f'Epoch {epoch+1} - Loss: {running_loss/total_batches:.4f}')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def run():
    pathlib.Path(_LOG_DIR).mkdir(exist_ok=True)
    
    for iteration in range(10):
        print(f'Starting iteration {iteration}')
        
        # Create model directory
        model_dir = pathlib.Path(_LOG_DIR) / str(iteration) / f'0_resnet50-{_LEARNING_RATE}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model, criterion, optimizer
        model = ResNet50Model().to(device)
        criterion = WeightedL1Loss(_CLASS_WEIGHTS)  # Use weighted loss
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders()
        
        # Calculate total steps for one cycle scheduler
        total_steps = len(train_loader) * 200  # 200 epochs
        
        # Training loop
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(200):  # Max epochs
            if epoch == 0:
                # First epoch: freeze base model
                model.freeze_base_model()
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=_LEARNING_RATE)
            elif epoch == 1:
                # After first epoch: unfreeze base model
                model.unfreeze_base_model()
                optimizer = optim.Adam(model.parameters(), lr=_LEARNING_RATE)
            
            # Create scheduler for this epoch
            scheduler = OneCycleScheduler(
                optimizer,
                max_lr=_LEARNING_RATE,
                total_steps=len(train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4
            )
            
            # Train
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
            
            # Validate
            val_loss = validate(model, val_loader, criterion)
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Save checkpoint if best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = model_dir / f'cp-{epoch+1:04d}-{val_loss:.4f}.pt'
                
                # Save regular checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, str(checkpoint_path))
                
                # Save TorchScript version
                model.eval()  # Set to evaluation mode
                scripted_model = torch.jit.script(model)
                script_path = model_dir / f'cp-{epoch+1:04d}-{val_loss:.4f}_scripted.pt'
                scripted_model.save(str(script_path))
                
                print(f'Saved checkpoint: {checkpoint_path}')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

if __name__ == '__main__':
    run()
