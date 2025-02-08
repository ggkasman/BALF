import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import os
import pathlib

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device('cpu')

# Paths
_TEST_DIR = '../Data/Testing/'
_DATA_SELECTOR = '*.png'
_MODEL_DIR = '../Model/'
_MODEL_FILE = 'resnet50-cp-0171-0.0967.pt'  # Changed to .pt extension

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target = dict([*self.model.named_modules()])[target_layer]
        target.register_forward_hook(self.save_activation)
        target.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, target_class):
        # Forward pass
        model_output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        
        # Get weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

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

def save_gradcam(image_path, heatmap, output_path, alpha=0.4):
    # Load and resize image
    img = cv2.imread(str(image_path))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(str(output_path), superimposed_img)

def run():
    # Initialize model
    model = ResNet50Model().to(device)
    model.eval()
    
    # Load model weights if available
    model_path = pathlib.Path(_MODEL_DIR) / _MODEL_FILE
    if model_path.exists():
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    
    # Initialize GradCAM with the last convolutional layer
    grad_cam = GradCAM(model, 'resnet.layer4.2.conv3')
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each image
    for case in pathlib.Path(_TEST_DIR).rglob(_DATA_SELECTOR):
        print(f"Processing image: {case}")
        
        # Load and preprocess image
        img = Image.open(case).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate GradCAM for each cell type
        for cell_type in range(4):
            try:
                heatmap = grad_cam.generate_heatmap(input_tensor, cell_type)
                output_path = str(case).replace('.png', f'_gradcam_{cell_type}.png')
                print(f"Saving visualization to: {output_path}")
                save_gradcam(case, heatmap, output_path)
            except Exception as e:
                print(f"Error processing cell type {cell_type}: {e}")

if __name__ == '__main__':
    run()
