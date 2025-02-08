import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save the initial learning rate
        self.init_lr = optimizer.param_groups[0]['lr']
        self.history = {'lr': [], 'loss': []}
        
    def range_test(self, train_loader, end_lr=10, num_iter=100, step_mode='exp'):
        # Save model state
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        optim_state = self.optimizer.state_dict()
        
        # Set up learning rate schedule
        if step_mode == 'exp':
            lr_schedule = np.geomspace(self.init_lr, end_lr, num=num_iter)
        else:
            lr_schedule = np.linspace(self.init_lr, end_lr, num=num_iter)
        
        iterator = iter(train_loader)
        for lr in lr_schedule:
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Record the loss
            self.history['lr'].append(lr)
            self.history['loss'].append(loss.item())
        
        # Restore model to initial state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        
    def plot(self, skip_start=10, skip_end=5):
        """
        Plot the learning rate vs loss, skipping the first few and last few points
        """
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:]
        
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()
        
    def suggest_lr(self, skip_start=10, skip_end=5):
        """
        Suggest a learning rate by picking point with steepest negative gradient
        """
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:]
        
        # Calculate gradients
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        
        return lrs[min_grad_idx]
