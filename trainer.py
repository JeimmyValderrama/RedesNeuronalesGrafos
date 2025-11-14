"""
trainer_smart.py
================
Entrenador que usa SmartMSTLoss para entrenar con criterios inteligentes.
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import config
from model_gat import EdgeGAT
from smart_loss import SmartMSTLoss


class EarlyStopping:
    """Early stopping para detener entrenamiento cuando validación no mejora."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class MST_Trainer:
    """Entrenador con criterio de decisión inteligente."""
    
    def __init__(self, 
                 model: EdgeGAT,
                 train_data: List,
                 val_data: List,
                 use_smart_loss: bool = True,
                 device: torch.device = None):
        """
        Args:
            model: Modelo GAT a entrenar
            train_data: Lista de grafos de entrenamiento
            val_data: Lista de grafos de validación
            use_smart_loss: Si usar SmartMSTLoss (True) o BCE estándar (False)
            device: Dispositivo (CPU/GPU)
        """
        self.device = device or config.DEVICE
        self.model = model.to(self.device)
        
        # Crear DataLoaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=False
        )
        
        # Optimizador (se crea después de inicializar modelo)
        self.optimizer = None
        self.lr = config.TRAINING_CONFIG['learning_rate']
        self.weight_decay = config.TRAINING_CONFIG['weight_decay']
        
        # Loss function
        self.use_smart_loss = use_smart_loss
        
        if use_smart_loss:
            self.criterion = SmartMSTLoss(alpha=config.ALPHA).to(self.device)
        else:
            pos_weight = torch.tensor([3.0]).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print("⚠️ Usando BCELoss estándar (solo distancia)")
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.TRAINING_CONFIG['early_stopping_patience']
        )
        
        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_path = config.MODELS_DIR / 'best_model_smart.pt'
    
    def _initialize_optimizer(self):
        """Inicializa optimizador después de que el modelo esté listo."""
        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            print(f"✓ Optimizador inicializado (lr={self.lr}, weight_decay={self.weight_decay})")
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Entrena una época completa.
        
        Returns:
            Tupla (loss, accuracy, f1_score)
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            
            # Inicializar optimizador en primer batch
            if self.optimizer is None:
                # Forward pass para inicializar modelo
                _ = self.model(batch)
                self._initialize_optimizer()
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch)
            
            # Loss inteligente o estándar
            if self.use_smart_loss:
                loss = self.criterion(logits, batch)
            else:
                loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item() * batch.num_graphs
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = total_correct / total_samples
        f1 = self._compute_f1(all_labels, all_preds)
        
        return avg_loss, accuracy, f1
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """
        Valida el modelo en conjunto de validación.
        
        Returns:
            Tupla (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            
            logits = self.model(batch)
            
            if self.use_smart_loss:
                loss = self.criterion(logits, batch)
            else:
                loss = self.criterion(logits, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = total_correct / total_samples
        f1 = self._compute_f1(all_labels, all_preds)
        
        return avg_loss, accuracy, f1
    
    def _compute_f1(self, labels: List, preds: List) -> float:
        """Calcula F1-score."""
        labels = np.array(labels)
        preds = np.array(preds)
        
        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return f1
    
    def train(self, num_epochs: int = None) -> Dict:
        """
        Entrenamiento completo del modelo.
        
        Args:
            num_epochs: Número de épocas (usa config si es None)
        
        Returns:
            Diccionario con historial de entrenamiento
        """
        num_epochs = num_epochs or config.TRAINING_CONFIG['num_epochs']
        log_interval = config.TRAINING_CONFIG['log_interval']
        
        print("=" * 70)
        print("INICIANDO ENTRENAMIENTO")
        print("=" * 70)
        print(f"Dispositivo: {self.device}")
        print(f"Épocas: {num_epochs}")
        print(f"Batch size: {config.TRAINING_CONFIG['batch_size']}")
        print(f"Learning rate: {self.lr}")
        print(f"Grafos de entrenamiento: {len(self.train_loader.dataset)}")
        print(f"Grafos de validación: {len(self.val_loader.dataset)}")
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Entrenar
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # Validar
            val_loss, val_acc, val_f1 = self.validate()
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Logging
            if epoch % log_interval == 0 or epoch == 1:
                print(f"\nÉpoca {epoch}/{num_epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Guardar mejor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if config.TRAINING_CONFIG['save_best_model']:
                    self.save_model(self.best_model_path)
                    if epoch % log_interval == 0 or epoch == 1:
                        print(f"  ✓ Mejor modelo guardado (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n⚠ Early stopping en época {epoch}")
                break
        
        print("\n" + "=" * 70)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 70)
        print(f"Mejor val_loss: {self.best_val_loss:.4f}")
        print(f"Modelo guardado en: {self.best_model_path}")
        
        return self.history
    
    def save_model(self, path: Path):
        """Guarda el modelo y metadatos."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'config': config.MODEL_CONFIG,
            'use_smart_loss': self.use_smart_loss,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path: Path):
        """Carga el modelo desde checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.use_smart_loss = checkpoint.get('use_smart_loss', False)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ Modelo cargado desde: {path}")
        print(f"  • Mejor val_loss: {self.best_val_loss:.4f}")
        print(f"  • Loss type: {'Smart' if self.use_smart_loss else 'Standard'}")
    
    def plot_history(self, save_path: Path = None):
        """Grafica el historial de entrenamiento."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Pérdida durante el entrenamiento', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title('Precisión durante el entrenamiento', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1-Score
        axes[2].plot(epochs, self.history['train_f1'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, self.history['val_f1'], 'r-', label='Val', linewidth=2)
        axes[2].set_xlabel('Época', fontsize=11)
        axes[2].set_ylabel('F1-Score', fontsize=11)
        axes[2].set_title('F1-Score durante el entrenamiento', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            print(f"✓ Gráfica guardada en: {save_path}")
        
        plt.show()