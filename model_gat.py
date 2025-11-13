"""
model_gat.py - VERSIÓN CORREGIDA
==================================
Acepta dimensión de entrada dinámica según características del dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import config


class EdgeGAT(nn.Module):
    """
    Graph Attention Network para predicción de MST.
    Ahora acepta dimensión de entrada dinámica.
    """
    
    def __init__(self, 
                 in_channels: int = None,  # 🔥 Ahora puede ser None
                 hidden_channels: int = None,
                 num_heads: int = None,
                 num_layers: int = None,
                 dropout: float = None):
        """
        Args:
            in_channels: Dimensión de entrada (None = se detecta automáticamente)
        """
        super(EdgeGAT, self).__init__()
        
        # Usar valores de config si no se especifican
        self.in_channels = in_channels  # Puede ser None temporalmente
        hidden_channels = hidden_channels or config.MODEL_CONFIG['hidden_channels']
        num_heads = num_heads or config.MODEL_CONFIG['num_heads']
        num_layers = num_layers or config.MODEL_CONFIG['num_layers']
        dropout = dropout or config.MODEL_CONFIG['dropout']
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # Las capas se crearán dinámicamente en el primer forward
        self.convs = None
        self.batch_norms = None
        self.edge_mlp = None
        self._initialized = False
    
    def _initialize_layers(self, in_channels: int):
        """
        Inicializa las capas con la dimensión de entrada correcta.
        Se llama automáticamente en el primer forward.
        """
        if self._initialized:
            return
        
        self.in_channels = in_channels
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Primera capa
        self.convs.append(
            GATConv(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                heads=self.num_heads,
                dropout=self.dropout,
                concat=True
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels * self.num_heads))
        
        # Capas intermedias
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=self.hidden_channels * self.num_heads,
                    out_channels=self.hidden_channels,
                    heads=self.num_heads,
                    dropout=self.dropout,
                    concat=True
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels * self.num_heads))
        
        # Última capa
        self.convs.append(
            GATConv(
                in_channels=self.hidden_channels * self.num_heads,
                out_channels=self.hidden_channels,
                heads=self.num_heads,
                dropout=self.dropout,
                concat=False
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        # MLP para clasificación de aristas
        edge_mlp_input = self.hidden_channels * 3 + 1
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input, self.hidden_channels * 2),
            nn.LeakyReLU(config.MODEL_CONFIG['negative_slope']),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.LeakyReLU(config.MODEL_CONFIG['negative_slope']),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 1)
        )
        
        self._initialized = True
    
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass con inicialización automática."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Inicializar capas si es necesario
        if not self._initialized:
            self._initialize_layers(x.shape[1])
        
        # Pasar por capas GAT
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x, negative_slope=config.MODEL_CONFIG['negative_slope'])
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Construir embeddings de aristas
        edge_embeddings = self._build_edge_embeddings(x, edge_index, edge_attr)
        
        # Clasificar aristas
        edge_logits = self.edge_mlp(edge_embeddings).squeeze(-1)
        
        return edge_logits
    
    def _build_edge_embeddings(self, 
                               node_embeddings: torch.Tensor,
                               edge_index: torch.Tensor,
                               edge_attr: torch.Tensor) -> torch.Tensor:
        """Construye embeddings de aristas."""
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]
        diff_embeddings = torch.abs(src_embeddings - dst_embeddings)
        
        edge_features = torch.cat([
            src_embeddings,
            dst_embeddings,
            diff_embeddings,
            edge_attr
        ], dim=1)
        
        return edge_features
    
    def predict_probabilities(self, data: Data) -> torch.Tensor:
        """Predice probabilidades de MST."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probabilities = torch.sigmoid(logits)
        return probabilities


def count_parameters(model: nn.Module) -> int:
    """Cuenta parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: EdgeGAT):
    """Imprime resumen del modelo."""
    print("=" * 70)
    print("RESUMEN DEL MODELO GAT")
    print("=" * 70)
    print(f"\nArquitectura:")
    print(f"  • Número de capas GAT: {model.num_layers}")
    print(f"  • Canales ocultos: {model.hidden_channels}")
    print(f"  • Cabezas de atención: {model.num_heads}")
    print(f"  • Dropout: {model.dropout}")
    
    if model._initialized:
        print(f"  • Dimensión de entrada: {model.in_channels}")
        total_params = count_parameters(model)
    else:
        print(f"\n⚠ Modelo no inicializado (se inicializará en primer forward)")
    
    print("=" * 70)