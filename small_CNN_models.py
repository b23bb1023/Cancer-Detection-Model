"""
Histopathologic Cancer Detection - PyTorch Implementation
GPU-enabled with CUDA support, AMP training, and Confidence-Based Analysis
Windows Compatible

FIXED VERSION:
- Proper data augmentation (applied dynamically per epoch)
- GPU prefetching for better utilization
- BatchNorm added to models
- Proper checkpoint saving (includes optimizer state)
- Removed duplicate caching
- Dynamic linear layer size calculation
"""

# ============================================
# IMPORTS
# ============================================
import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================
# CONFIGURATION
# ============================================
LABELS_PATH = 'train_labels.csv'
TRAIN_DIR = 'train'
TEST_DIR = 'test'

# Hyperparameters
BATCH_SIZE = 192  # Increased from 128 for better GPU utilization
LEARNING_RATE = 0.0001
EPOCHS = 15
IMAGE_SIZE = 96
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
USE_AMP = True

# ============================================
# CUSTOM DATASET CLASSES
# ============================================
class CancerDataset(Dataset):
    """Standard dataset - loads from disk each time"""
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'id'] + '.tif'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.loc[idx, 'label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


class CachedAugmentedDataset(Dataset):
    """
    Caches decoded images in RAM but applies transforms DYNAMICALLY per epoch.
    This is the key fix - augmentation now works properly!
    """
    def __init__(self, dataframe, img_dir, transform=None, cache_transform=None):
        """
        Args:
            dataframe: DataFrame with 'id' and 'label' columns
            img_dir: Directory containing images
            transform: Transforms to apply dynamically (including augmentation)
            cache_transform: Optional transform to apply during caching (e.g., resize only)
        """
        self.transform = transform
        self.images = []
        self.labels = []
        
        dataframe = dataframe.reset_index(drop=True)
        
        # Cache decoded images (without augmentation)
        print("Caching images to RAM...")
        for idx in tqdm(range(len(dataframe)), desc="Loading images"):
            img_name = dataframe.loc[idx, 'id'] + '.tif'
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            
            # Apply cache transform if provided (e.g., resize to save memory)
            if cache_transform:
                img = cache_transform(img)
            
            self.images.append(img)
            self.labels.append(dataframe.loc[idx, 'label'])
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(f"✓ Cached {len(self.images)} images to RAM")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Apply transforms DYNAMICALLY - this is where augmentation happens!
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


# ============================================
# GPU PREFETCHER FOR BETTER UTILIZATION
# ============================================
class GPUPrefetcher:
    """
    Prefetches batches to GPU asynchronously.
    Overlaps data transfer with computation for better GPU utilization.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream:
            self.preload()
        return self
    
    def preload(self):
        try:
            self.next_images, self.next_labels = next(self.iter)
        except StopIteration:
            self.next_images = None
            self.next_labels = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_images = self.next_images.to(self.device, non_blocking=True)
            self.next_labels = self.next_labels.to(self.device, non_blocking=True)
    
    def __next__(self):
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            images = self.next_images
            labels = self.next_labels
            if images is None:
                raise StopIteration
            self.preload()
            return images, labels
        else:
            # CPU fallback
            images, labels = next(self.iter)
            return images.to(self.device), labels.to(self.device)
    
    def __len__(self):
        return len(self.loader)


# ============================================
# MODEL DEFINITIONS (with BatchNorm)
# ============================================
class BaselineModel(nn.Module):
    """Simple CNN with 2 conv layers + BatchNorm"""
    def __init__(self, input_size=96):
        super(BaselineModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            flatten_size = self.features(dummy).view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdvancedModel(nn.Module):
    """Deeper CNN with 4 conv layers + BatchNorm + Dropout"""
    def __init__(self, input_size=96):
        super(AdvancedModel, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================
# TRAINING UTILITIES
# ============================================
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_weights = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
    
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print("Restored best model weights.")


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save complete checkpoint including optimizer state"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)


def load_checkpoint_if_exists(model, optimizer, checkpoint_path, device):
    """Load checkpoint with full optimizer state restoration"""
    if os.path.exists(checkpoint_path):
        print(f"✓ Found existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            val_acc = checkpoint.get('val_acc', 0)
            print(f"✓ Loaded checkpoint from epoch {start_epoch} (val_acc: {val_acc:.4f})")
            return True, start_epoch
        else:
            # Legacy checkpoint (just model weights)
            model.load_state_dict(checkpoint)
            print("✓ Loaded legacy checkpoint (model weights only)")
            return True, 0
    return False, 0


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with GPU prefetching"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use GPU prefetcher for better utilization
    prefetcher = GPUPrefetcher(train_loader, device)
    
    pbar = tqdm(prefetcher, desc='Training', leave=False, total=len(train_loader))
    for images, labels in pbar:
        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return running_loss / len(train_loader), correct / total


def validate_epoch(model, val_loader, criterion, device, use_amp=False):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    prefetcher = GPUPrefetcher(val_loader, device)
    
    with torch.no_grad():
        pbar = tqdm(prefetcher, desc='Validation', leave=False, total=len(val_loader))
        for images, labels in pbar:
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return running_loss / len(val_loader), correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name, checkpoint_path, use_amp=False):
    
    early_stopping = EarlyStopping(patience=5, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'learning_rate': []}
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, use_amp)
        
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"✓ Model saved to {checkpoint_path} (val_acc: {val_acc:.4f})")
        
        early_stopping(val_loss, model)
        scheduler.step(val_loss)
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best_weights(model)
            break
    
    return history


def get_predictions(model, data_loader, device):
    """Get predictions with probabilities and confidence scores"""
    model.eval()
    all_probs = []
    all_labels = []
    
    prefetcher = GPUPrefetcher(data_loader, device)
    
    with torch.no_grad():
        for images, labels in tqdm(prefetcher, desc='Predicting', total=len(data_loader)):
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Calculate confidence: how confident the model is in its prediction
    all_confidence = np.where(all_probs > 0.5, all_probs, 1 - all_probs)
    
    return all_labels, all_probs, all_preds, all_confidence


# ============================================
# METRICS AND ANALYSIS FUNCTIONS
# ============================================
def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': auc(*roc_curve(y_true, y_prob)[:2]),
        'Avg Precision': average_precision_score(y_true, y_prob)
    }


def calculate_metrics_for_subset(y_true, y_pred, y_prob):
    """Calculate metrics for a subset of predictions"""
    if len(y_true) == 0:
        return None
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases where only one class is present
    if cm.shape == (1, 1):
        if y_true[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'Total Samples': len(y_true),
        'Accuracy': accuracy_score(y_true, y_pred),
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
    }
    
    # Calculate rates safely
    metrics['Sensitivity (Recall)'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    if metrics['Precision'] + metrics['Sensitivity (Recall)'] > 0:
        metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Sensitivity (Recall)']) / \
                             (metrics['Precision'] + metrics['Sensitivity (Recall)'])
    else:
        metrics['F1-Score'] = 0.0
    
    metrics['Error Rate'] = 1 - metrics['Accuracy']
    
    return metrics


def analyze_confidence_tiers(y_true, y_pred, y_prob, confidence, model_name):
    """Analyze predictions by confidence tiers"""
    
    tiers = {
        'Low (<50%)': (0.0, 0.50),
        'Medium (50-90%)': (0.50, 0.90),
        'High (>90%)': (0.90, 1.01)
    }
    
    tier_metrics = {}
    
    print(f"\n{'='*60}")
    print(f"{model_name} - CONFIDENCE-BASED ANALYSIS")
    print(f"{'='*60}")
    
    for tier_name, (low, high) in tiers.items():
        mask = (confidence >= low) & (confidence < high)
        
        if mask.sum() == 0:
            print(f"\n{tier_name}: No samples in this tier")
            tier_metrics[tier_name] = None
            continue
        
        tier_y_true = y_true[mask]
        tier_y_pred = y_pred[mask]
        tier_y_prob = y_prob[mask]
        
        metrics = calculate_metrics_for_subset(tier_y_true, tier_y_pred, tier_y_prob)
        tier_metrics[tier_name] = metrics
        
        print(f"\n{tier_name}:")
        print(f"  Samples: {metrics['Total Samples']}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  Error Rate: {metrics['Error Rate']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Sensitivity (Recall)']:.4f}")
    
    return tier_metrics


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def show_samples(labels_df, img_dir, label, num_samples=5):
    """Display sample images for a specific label"""
    sample_df = labels_df[labels_df['label'] == label].sample(n=num_samples, random_state=42)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    title = 'Cancer Positive Samples' if label == 1 else 'Cancer Negative Samples'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img_path = os.path.join(img_dir, row['id'] + '.tif')
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"ID: {row['id'][:8]}...")
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Accuracy
    axes[0].plot(history['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0].plot(history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['loss'], 'b-', label='Train', linewidth=2)
    axes[1].plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(history['learning_rate'], 'g-', linewidth=2)
    axes[2].set_title(f'{model_name} - Learning Rate', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Cancer', 'Cancer'],
                yticklabels=['No Cancer', 'Cancer'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_confidence_analysis(tier_metrics_baseline, tier_metrics_advanced):
    """Plot confidence tier analysis comparison"""
    tiers = ['Low (<50%)', 'Medium (50-90%)', 'High (>90%)']
    
    baseline_acc = []
    advanced_acc = []
    baseline_err = []
    advanced_err = []
    
    for tier in tiers:
        if tier_metrics_baseline.get(tier):
            baseline_acc.append(tier_metrics_baseline[tier]['Accuracy'])
            baseline_err.append(tier_metrics_baseline[tier]['Error Rate'])
        else:
            baseline_acc.append(0)
            baseline_err.append(0)
        
        if tier_metrics_advanced.get(tier):
            advanced_acc.append(tier_metrics_advanced[tier]['Accuracy'])
            advanced_err.append(tier_metrics_advanced[tier]['Error Rate'])
        else:
            advanced_acc.append(0)
            advanced_err.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(tiers))
    width = 0.35
    
    # Accuracy by confidence tier
    axes[0].bar(x - width/2, baseline_acc, width, label='Baseline', color='#3498db')
    axes[0].bar(x + width/2, advanced_acc, width, label='Advanced', color='#e74c3c')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Confidence Tier', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tiers)
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Error rate by confidence tier
    axes[1].bar(x - width/2, baseline_err, width, label='Baseline', color='#3498db')
    axes[1].bar(x + width/2, advanced_err, width, label='Advanced', color='#e74c3c')
    axes[1].set_ylabel('Error Rate')
    axes[1].set_title('Error Rate by Confidence Tier', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tiers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_analysis_by_confidence(y_true, y_pred, confidence, model_name):
    """Plot error analysis by confidence bins"""
    bins = np.linspace(0.5, 1.0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    errors = y_true != y_pred
    error_rates = []
    sample_counts = []
    
    for i in range(len(bins) - 1):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() > 0:
            error_rates.append(errors[mask].mean())
            sample_counts.append(mask.sum())
        else:
            error_rates.append(0)
            sample_counts.append(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#e74c3c'
    ax1.bar(bin_centers, error_rates, width=0.04, color=color1, alpha=0.7, label='Error Rate')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Error Rate', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, max(error_rates) * 1.2 if max(error_rates) > 0 else 0.1])
    
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.plot(bin_centers, sample_counts, 'o-', color=color2, linewidth=2, markersize=8, label='Sample Count')
    ax2.set_ylabel('Sample Count', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'{model_name} - Error Rate by Confidence', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_error_by_confidence.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_predictions(model, data_loader, device, model_name, num_samples=10):
    """Show sample predictions"""
    model.eval()
    
    images, labels_batch = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels_batch = labels_batch[:num_samples]
    
    with torch.no_grad():
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
    
    images_np = images.cpu().numpy()
    images_np = (images_np * 0.5 + 0.5)
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images_np))):
        ax = axes[i]
        ax.imshow(np.clip(images_np[i], 0, 1))
        
        true_label = 'Cancer' if labels_batch[i] == 1 else 'No Cancer'
        pred_prob = probs[i]
        pred_label = 'Cancer' if pred_prob > 0.5 else 'No Cancer'
        confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
        
        color = 'green' if true_label == pred_label else 'red'
        
        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1%}',
                     fontsize=9, color=color)
        ax.axis('off')
    
    plt.suptitle(f'{model_name} - Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_misclassifications(y_true, y_pred, model_name):
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    false_positives = misclassified_idx[y_pred[misclassified_idx] == 1]
    false_negatives = misclassified_idx[y_pred[misclassified_idx] == 0]
    
    print(f"\n{model_name} - Misclassification Analysis:")
    print(f"  Total Misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_true)*100:.2f}%)")
    print(f"  False Positives (Predicted Cancer, Actually Healthy): {len(false_positives)}")
    print(f"  False Negatives (Predicted Healthy, Actually Cancer): {len(false_negatives)}")
    print(f"\n  ⚠️ False Negatives are more critical in cancer detection!")
    
    return len(false_positives), len(false_negatives)


def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


# ============================================
# MAIN FUNCTION
# ============================================
def main():
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print("DEVICE CONFIGURATION")
    print("=" * 50)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"cuDNN Benchmark: Enabled")
        print(f"Mixed Precision (AMP): {USE_AMP}")
    else:
        print("CUDA is not available. Running on CPU.")
    print()

    # Load labels
    labels = pd.read_csv(LABELS_PATH)

    # Dataset overview
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"\nShape: {labels.shape}")
    print(f"\nColumn Types:\n{labels.dtypes}")
    print(f"\nMissing Values:\n{labels.isnull().sum()}")
    print(f"\nClass Distribution:\n{labels['label'].value_counts()}")
    print(f"\nClass Balance: {labels['label'].value_counts(normalize=True).round(4) * 100}%")
    print("\nFirst 5 rows:")
    print(labels.head())
    print()

    # Sample visualization
    show_samples(labels, TRAIN_DIR, label=0)
    show_samples(labels, TRAIN_DIR, label=1)

    # Class distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    class_counts = labels['label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    bars = axes[0].bar(['No Cancer (0)', 'Cancer (1)'], class_counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_xlabel('Class', fontsize=12)
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                     f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    axes[1].pie(class_counts.values, labels=['No Cancer (0)', 'Cancer (1)'],
                autopct='%1.1f%%', colors=colors, explode=(0.02, 0.02),
                shadow=True, startangle=90)
    axes[1].set_title('Class Percentage Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nClass Imbalance Ratio: 1:{class_counts[0]/class_counts[1]:.2f}")

    # Image statistics
    num_train_images = len(os.listdir(TRAIN_DIR))
    num_test_images = len(os.listdir(TEST_DIR))

    print("=" * 50)
    print("IMAGE STATISTICS")
    print("=" * 50)
    print(f"Training Images: {num_train_images:,}")
    print(f"Test Images: {num_test_images:,}")
    print(f"Total Images: {num_train_images + num_test_images:,}")
    print(f"\nImage Format: .tif")
    print(f"Image Size: 96x96 pixels")
    print(f"Color Channels: 3 (RGB)")
    print()

    # Split data
    train_labels_df, val_labels_df = train_test_split(
        labels, test_size=0.2, stratify=labels['label'], random_state=17
    )
    print(f"Training samples: {len(train_labels_df)}")
    print(f"Validation samples: {len(val_labels_df)}")

    # ============================================
    # DATA TRANSFORMS - IMPORTANT!
    # ============================================
    # Cache transform: Applied ONCE during caching (resize only)
    cache_transform = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Train transform: Applied DYNAMICALLY each epoch (includes augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Validation transform: No augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ============================================
    # CREATE DATASETS WITH PROPER CACHING
    # ============================================
    print("\n" + "=" * 50)
    print("LOADING AND CACHING DATA")
    print("=" * 50)
    
    # Training dataset: Cache decoded images, apply augmentation dynamically
    train_dataset = CachedAugmentedDataset(
        train_labels_df, 
        TRAIN_DIR, 
        transform=train_transform,
        cache_transform=cache_transform  # Only resize during caching
    )
    
    # Validation dataset: Standard loading (no caching needed, no augmentation)
    val_dataset = CancerDataset(val_labels_df, TRAIN_DIR, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        
        drop_last=True  # Helps with BatchNorm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"Mixed Precision (AMP): {USE_AMP}")
    print()

    # Create models
    baseline_model = BaselineModel(input_size=IMAGE_SIZE).to(device)
    advanced_model = AdvancedModel(input_size=IMAGE_SIZE).to(device)

    print("=" * 50)
    print("BASELINE MODEL ARCHITECTURE")
    print("=" * 50)
    print(baseline_model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    print()

    print("=" * 50)
    print("ADVANCED MODEL ARCHITECTURE")
    print("=" * 50)
    print(advanced_model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in advanced_model.parameters()):,}")
    print()

    # ============================================
    # TRAIN BASELINE MODEL
    # ============================================
    print("=" * 50)
    print("TRAINING BASELINE MODEL")
    print("=" * 50)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)

    baseline_ckpt = 'baseline_model_best.pth'
    resumed, start_epoch = load_checkpoint_if_exists(
        baseline_model, optimizer_baseline, baseline_ckpt, device
    )

    history_baseline = train_model(
        baseline_model, train_loader, val_loader, criterion, optimizer_baseline,
        EPOCHS, device, 'Baseline Model', baseline_ckpt, use_amp=USE_AMP
    )

    # ============================================
    # TRAIN ADVANCED MODEL
    # ============================================
    print("\n" + "=" * 50)
    print("TRAINING ADVANCED MODEL")
    print("=" * 50)

    optimizer_advanced = optim.Adam(advanced_model.parameters(), lr=LEARNING_RATE)

    advanced_ckpt = 'advanced_model_best.pth'
    resumed, start_epoch = load_checkpoint_if_exists(
        advanced_model, optimizer_advanced, advanced_ckpt, device
    )

    history_advanced = train_model(
        advanced_model, train_loader, val_loader, criterion, optimizer_advanced,
        EPOCHS, device, 'Advanced Model', advanced_ckpt, use_amp=USE_AMP
    )

    # Plot training history
    plot_training_history(history_baseline, 'Baseline Model')
    plot_training_history(history_advanced, 'Advanced Model')

    # ============================================
    # GENERATE PREDICTIONS
    # ============================================
    print("=" * 50)
    print("GENERATING PREDICTIONS")
    print("=" * 50)

    # Load best weights
    baseline_checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=False)
    if isinstance(baseline_checkpoint, dict) and 'model_state_dict' in baseline_checkpoint:
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(baseline_checkpoint)

    advanced_checkpoint = torch.load(advanced_ckpt, map_location=device, weights_only=False)
    if isinstance(advanced_checkpoint, dict) and 'model_state_dict' in advanced_checkpoint:
        advanced_model.load_state_dict(advanced_checkpoint['model_state_dict'])
    else:
        advanced_model.load_state_dict(advanced_checkpoint)

    y_true, y_prob_baseline, y_pred_baseline, confidence_baseline = get_predictions(baseline_model, val_loader, device)
    _, y_prob_advanced, y_pred_advanced, confidence_advanced = get_predictions(advanced_model, val_loader, device)

    print(f"\n✓ Predictions generated for {len(y_true)} validation samples")

    # ============================================
    # EVALUATION
    # ============================================
    
    # Confusion matrices
    print("=" * 50)
    print("CONFUSION MATRICES")
    print("=" * 50)
    cm_baseline = plot_confusion_matrix(y_true, y_pred_baseline, 'Baseline Model')
    cm_advanced = plot_confusion_matrix(y_true, y_pred_advanced, 'Advanced Model')

    # Classification reports
    print("=" * 50)
    print("CLASSIFICATION REPORTS")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("BASELINE MODEL - Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred_baseline,
                                target_names=['No Cancer (0)', 'Cancer (1)']))

    print("\n" + "=" * 50)
    print("ADVANCED MODEL - Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred_advanced,
                                target_names=['No Cancer (0)', 'Cancer (1)']))

    # Confidence-based analysis
    print("\n" + "=" * 60)
    print("CONFIDENCE-BASED ANALYSIS")
    print("=" * 60)
    print("\nThis analysis shows model reliability at different confidence levels.")
    print("Higher confidence predictions should have lower error rates.\n")

    tier_metrics_baseline = analyze_confidence_tiers(
        y_true, y_pred_baseline, y_prob_baseline, confidence_baseline, 'Baseline Model'
    )
    tier_metrics_advanced = analyze_confidence_tiers(
        y_true, y_pred_advanced, y_prob_advanced, confidence_advanced, 'Advanced Model'
    )

    # Plot confidence analysis
    plot_confidence_analysis(tier_metrics_baseline, tier_metrics_advanced)
    plot_error_analysis_by_confidence(y_true, y_pred_baseline, confidence_baseline, 'Baseline Model')
    plot_error_analysis_by_confidence(y_true, y_pred_advanced, confidence_advanced, 'Advanced Model')

    # ROC curves
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, y_prob_baseline)
    auc_baseline = auc(fpr_baseline, tpr_baseline)
    fpr_advanced, tpr_advanced, _ = roc_curve(y_true, y_prob_advanced)
    auc_advanced = auc(fpr_advanced, tpr_advanced)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_baseline, tpr_baseline, 'b-', linewidth=2,
             label=f'Baseline Model (AUC = {auc_baseline:.4f})')
    plt.plot(fpr_advanced, tpr_advanced, 'r-', linewidth=2,
             label=f'Advanced Model (AUC = {auc_advanced:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nAUC Scores:")
    print(f"  Baseline Model: {auc_baseline:.4f}")
    print(f"  Advanced Model: {auc_advanced:.4f}")

    # Precision-Recall curves
    precision_baseline, recall_baseline, _ = precision_recall_curve(y_true, y_prob_baseline)
    precision_advanced, recall_advanced, _ = precision_recall_curve(y_true, y_prob_advanced)
    ap_baseline = average_precision_score(y_true, y_prob_baseline)
    ap_advanced = average_precision_score(y_true, y_prob_advanced)

    plt.figure(figsize=(10, 8))
    plt.plot(recall_baseline, precision_baseline, 'b-', linewidth=2,
             label=f'Baseline Model (AP = {ap_baseline:.4f})')
    plt.plot(recall_advanced, precision_advanced, 'r-', linewidth=2,
             label=f'Advanced Model (AP = {ap_advanced:.4f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nAverage Precision Scores:")
    print(f"  Baseline Model: {ap_baseline:.4f}")
    print(f"  Advanced Model: {ap_advanced:.4f}")

    # Model comparison summary
    metrics_baseline = calculate_metrics(y_true, y_pred_baseline, y_prob_baseline)
    metrics_advanced = calculate_metrics(y_true, y_pred_advanced, y_prob_advanced)

    comparison_df = pd.DataFrame({
        'Metric': list(metrics_baseline.keys()),
        'Baseline Model': [f"{v:.4f}" for v in metrics_baseline.values()],
        'Advanced Model': [f"{v:.4f}" for v in metrics_advanced.values()]
    })

    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    # Visual comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_baseline))
    width = 0.35

    bars1 = ax.bar(x - width/2, list(metrics_baseline.values()), width, label='Baseline Model', color='#3498db')
    bars2 = ax.bar(x + width/2, list(metrics_advanced.values()), width, label='Advanced Model', color='#e74c3c')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_baseline.keys()), rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Sample predictions
    print("=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    show_predictions(baseline_model, val_loader, device, 'Baseline Model')
    show_predictions(advanced_model, val_loader, device, 'Advanced Model')

    # Misclassification analysis
    fp_baseline, fn_baseline = analyze_misclassifications(y_true, y_pred_baseline, 'Baseline Model')
    fp_advanced, fn_advanced = analyze_misclassifications(y_true, y_pred_advanced, 'Advanced Model')

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, [fp_baseline, fn_baseline], width, label='Baseline Model', color='#3498db')
    bars2 = ax.bar(x + width/2, [fp_advanced, fn_advanced], width, label='Advanced Model', color='#e74c3c')

    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Misclassification Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['False Positives\n(Healthy → Cancer)', 'False Negatives\n(Cancer → Healthy)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('misclassification_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ============================================
    # SAVE MODELS AND RESULTS
    # ============================================
    print("=" * 50)
    print("SAVING MODELS AND RESULTS")
    print("=" * 50)

    torch.save(baseline_model.state_dict(), 'baseline_model_final.pth')
    torch.save(advanced_model.state_dict(), 'advanced_model_final.pth')
    print("✓ Models saved successfully!")

    # Save all results including confidence analysis
    results = {
        'baseline_model': {
            'overall_metrics': metrics_baseline,
            'confidence_tier_metrics': {k: v for k, v in tier_metrics_baseline.items() if v is not None}
        },
        'advanced_model': {
            'overall_metrics': metrics_advanced,
            'confidence_tier_metrics': {k: v for k, v in tier_metrics_advanced.items() if v is not None}
        },
        'training_params': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'image_size': f'{IMAGE_SIZE}x{IMAGE_SIZE}',
            'train_samples': len(train_labels_df),
            'val_samples': len(val_labels_df),
            'device': str(device),
            'amp_enabled': USE_AMP
        }
    }

    results = convert_to_serializable(results)

    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("✓ Results saved to model_results.json")

    comparison_df.to_csv('model_comparison.csv', index=False)
    print("✓ Comparison table saved to model_comparison.csv")

    # Save confidence analysis to CSV
    confidence_data = []
    for tier_name in ['Low (<50%)', 'Medium (50-90%)', 'High (>90%)']:
        if tier_metrics_baseline.get(tier_name):
            row = {'Model': 'Baseline', 'Confidence Tier': tier_name}
            row.update(tier_metrics_baseline[tier_name])
            confidence_data.append(row)
        if tier_metrics_advanced.get(tier_name):
            row = {'Model': 'Advanced', 'Confidence Tier': tier_name}
            row.update(tier_metrics_advanced[tier_name])
            confidence_data.append(row)
    
    confidence_df = pd.DataFrame(confidence_data)
    confidence_df.to_csv('confidence_analysis.csv', index=False)
    print("✓ Confidence analysis saved to confidence_analysis.csv")

    print("\n" + "=" * 50)
    print("FILES GENERATED:")
    print("=" * 50)
    print("  📊 Figures:")
    print("     - class_distribution.png")
    print("     - baseline_model_training_history.png")
    print("     - advanced_model_training_history.png")
    print("     - baseline_model_confusion_matrix.png")
    print("     - advanced_model_confusion_matrix.png")
    print("     - roc_curve_comparison.png")
    print("     - precision_recall_curve.png")
    print("     - model_comparison.png")
    print("     - baseline_model_predictions.png")
    print("     - advanced_model_predictions.png")
    print("     - misclassification_comparison.png")
    print("     - confidence_analysis.png")
    print("     - baseline_model_error_by_confidence.png")
    print("     - advanced_model_error_by_confidence.png")
    print("  📁 Models:")
    print("     - baseline_model_best.pth")
    print("     - advanced_model_best.pth")
    print("     - baseline_model_final.pth")
    print("     - advanced_model_final.pth")
    print("  📄 Data:")
    print("     - model_results.json")
    print("     - model_comparison.csv")
    print("     - confidence_analysis.csv")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)

    # Print confidence analysis summary
    print("\n" + "=" * 60)
    print("CONFIDENCE RELIABILITY SUMMARY")
    print("=" * 60)
    print("\nInterpretation Guide:")
    print("-" * 60)
    print("• Low Confidence (<50%): Model is uncertain - expect higher errors")
    print("• Medium Confidence (50-90%): Moderate certainty")
    print("• High Confidence (>90%): Model is very certain - expect fewer errors")
    print("\nKey Insight: If high confidence predictions have low error rates,")
    print("the model's confidence is well-calibrated and trustworthy.")
    print("=" * 60)


# ============================================
# ENTRY POINT
# ============================================
if __name__ == '__main__':
    main()