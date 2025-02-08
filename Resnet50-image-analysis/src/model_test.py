import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device('cpu')

# Paths
_VAL_DIR = '../Data/Validation/'
_TEST_DIR = '../Data/Testing/'
_DATA_SELECTOR = '*.png'
_MODEL_DIR = '../Model/'
_MODEL_SELECTOR = '*.pt'  # PyTorch model files
_BATCH_SIZE = 1

class CellDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = pathlib.Path(data_dir)
        self.transform = transform
        self.image_paths = list(self.data_dir.rglob(_DATA_SELECTOR))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
            
        print(f"Found {len(self.image_paths)} images in {data_dir}")
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load labels - fix the label path construction
        label_path = str(img_path).replace('.png', '_labels.txt')
        try:
            with open(label_path, 'r') as f:
                labels = [float(x) for x in f.read().strip().split(',')]
        except FileNotFoundError:
            # Try alternative label file name
            label_path = str(img_path.parent / 'labels.txt')
            with open(label_path, 'r') as f:
                labels = [float(x) for x in f.read().strip().split(',')]
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels, str(img_path)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders():
    transform = get_transform()
    
    val_dataset = CellDataset(_VAL_DIR, transform=transform)
    test_dataset = CellDataset(_TEST_DIR, transform=transform)
    
    val_loader = DataLoader(val_dataset, batch_size=_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=_BATCH_SIZE, shuffle=False, num_workers=0)
    
    return val_loader, test_loader

def evaluate_model(model, dataloader, session_dir, subset='val'):
    model.eval()
    criterion = nn.L1Loss(reduction='none')
    
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Calculate loss for each sample
            losses = criterion(outputs, labels)
            mean_loss = losses.mean().item()
            
            # Save predictions
            for j, path in enumerate(paths):
                output_file = session_dir / f'{subset}-{i}-{mean_loss:.4f}.txt'
                with open(output_file, 'w') as f:
                    f.writelines([
                        f'Neutrophils: {outputs[j,0]:.1%} estimated vs. {labels[j,0]:.1%}\n',
                        f'Eosinophils: {outputs[j,1]:.1%} estimated vs. {labels[j,1]:.1%}\n',
                        f'Lymphocytes: {outputs[j,2]:.1%} estimated vs. {labels[j,2]:.1%}\n',
                        f'Macrophages: {outputs[j,3]:.1%} estimated vs. {labels[j,3]:.1%}\n'
                    ])
                print(f'Saved predictions to: {output_file}')

def run():
    val_loader, test_loader = create_dataloaders()
    
    for iteration in range(10):
        print(f'Processing iteration {iteration}')
        session_dir = pathlib.Path(_MODEL_DIR) / str(iteration) / f'0_resnet50-1e-5'
        
        if not session_dir.exists():
            print(f'Directory not found: {session_dir}')
            continue
            
        # Find best model
        model_files = list(session_dir.glob(_MODEL_SELECTOR))
        if not model_files:
            print(f'No model files found in {session_dir}')
            continue
            
        # Sort by validation loss (extracted from filename)
        model_files.sort(key=lambda x: float(str(x).split('-')[-1].replace('.pt', '')))
        best_model_path = model_files[0]
        loss_value = float(str(best_model_path).split('-')[-1].replace('.pt', ''))
        
        print(f'Best model: {best_model_path} (loss: {loss_value})')
        if loss_value > 0.1:
            print(f'Skipping model with loss > 0.1')
            continue
            
        # Load model
        checkpoint = torch.load(best_model_path)
        model = ResNet50Model().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate on validation and test sets
        print('Evaluating validation set...')
        evaluate_model(model, val_loader, session_dir, 'val')
        
        print('Evaluating test set...')
        evaluate_model(model, test_loader, session_dir, 'test')

class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    run()
