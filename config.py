"""
config.py - VERSIÓN COMPLETA CON TODAS LAS CARACTERÍSTICAS
===========================================================
"""

import os
from pathlib import Path
import torch

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

ALPHA = 0.7  # Balance para SmartMSTLoss

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# PARÁMETROS DEL GRAFO REAL (TIBASOSA)
# ============================================================================
GRAPH_CONFIG = {
    'num_nodes': 17,
    'num_edges': None,
    'coordinates_file': 'tibasosa_coordenadas.csv',
    'graph_file': 'tibasosa_graph.csv',
    'use_complete_graph': True,
    'max_distance_meters': None,
    'use_proximity_graph': False,
    'k_neighbors': 5,
}

# ============================================================================
# PARÁMETROS DEL DATASET SINTÉTICO
# ============================================================================
DATASET_CONFIG = {
    'num_graphs': 1000,        # Suficientes grafos para aprender
    'min_nodes': 12,
    'max_nodes': 25,
    'min_edges': 20,
    'max_edges': 50,
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    'seed': 42,
}

# ============================================================================
# PARÁMETROS DEL MODELO GNN (GAT)
# ============================================================================
MODEL_CONFIG = {
    'hidden_channels': 128,
    'num_heads': 6,
    'num_layers': 4,
    'dropout': 0.3,
    'activation': 'leaky_relu',
    'negative_slope': 0.2,
}

# ============================================================================
# PARÁMETROS DE ENTRENAMIENTO
# ============================================================================
TRAINING_CONFIG = {
    'learning_rate': 0.0005,       # MÁS LENTO para aprender mejor
    'weight_decay': 1e-4,
    'batch_size': 32,
    'num_epochs': 100,             # MÁS ÉPOCAS
    'early_stopping_patience': 60, # MÁS PACIENCIA
    'save_best_model': True,
    'log_interval': 10,
}

# ============================================================================
# PARÁMETROS DE EVALUACIÓN
# ============================================================================
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'optimality_gap'],
    'compare_algorithms': ['kruskal', 'prim', 'gnn_guided'],
}

# ============================================================================
# PARÁMETROS DE VISUALIZACIÓN
# ============================================================================
VISUALIZATION_CONFIG = {
    'figure_size': (14, 10),
    'node_size': 600,
    'node_color': '#1f77b4',
    'edge_width': 2.5,
    'font_size': 9,
    'dpi': 300,
    'save_format': 'png',
}

# ============================================================================
# COORDENADAS GEOGRÁFICAS DE TIBASOSA
# ============================================================================
TIBASOSA_BOUNDS = {
    'lat_min': 5.741,
    'lat_max': 5.753,
    'lon_min': -73.008,
    'lon_max': -72.993,
    'center_lat': 5.747,
    'center_lon': -73.001,
}

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'training.log',
}

# ============================================================================
# CONFIGURACIÓN DE DISPOSITIVO (GPU/CPU)
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo configurado: {DEVICE}")

# ============================================================================
# PESOS PARA DECISIÓN DEL GNN - ¡TODAS LAS CARACTERÍSTICAS!
# ============================================================================
# IMPORTANTE: Reducir distancia para dar más peso a características inteligentes
FEATURE_WEIGHTS = {
    # DISTANCIA (reducida significativamente)
    'distance': 0.15,              # 15% - Distancia física (ANTES 25%)
    
    # CARACTERÍSTICAS NUMÉRICAS (50% total)
    'capacity': 0.09,              # 9% - Capacidad kVA
    'demand': 0.13,                # 13% - Demanda proyectada kVA ⭐
    'cost': 0.08,                  # 8% - Costo de mantenimiento anual
    'installation_time': 0.07,     # 7% - Tiempo de instalación (días)
    'risk': 0.13,                  # 13% - Factor de riesgo (invertido) ⭐
    
    # CARACTERÍSTICAS CATEGÓRICAS (35% total)
    'type': 0.07,                  # 7% - Tipo (residencial/comercial/público) 🔥
    'priority': 0.08,              # 8% - Prioridad (alta/media)
    'criticality': 0.18,           # 18% - Criticidad ⭐⭐⭐ MÁS IMPORTANTE
    'zone': 0.04,                  # 4% - Zona (urbana/comercial/recreativa/educativa)
    'accessibility': 0.04,         # 4% - Accesibilidad (fácil/media/difícil)
    'land_availability': 0.02,     # 2% - Disponibilidad de terreno
}

# ============================================================================
# MAPEO DE CARACTERÍSTICAS A NOMBRES ALTERNATIVOS
# ============================================================================
FEATURE_ALIASES = {
    'capacity': ['capacidad', 'capacidad_kva', 'kva'],
    'demand': ['demanda', 'demanda_proyectada', 'demanda_proyectada_kva'],
    'cost': ['costo', 'costo_mantenimiento', 'costo_mantenimiento_anual'],
    'installation_time': ['tiempo', 'tiempo_instalacion', 'tiempo_instalacion_dias'],
    'risk': ['riesgo', 'factor_riesgo', 'factor'],
    'type': ['tipo', 'category', 'categoria'],
    'priority': ['prioridad', 'priority'],
    'criticality': ['criticidad', 'criticalidad'],
    'zone': ['zona', 'zone', 'area'],
    'accessibility': ['accesibilidad', 'acceso', 'access'],
    'land_availability': ['disponibilidad', 'disponibilidad_terreno', 'terreno'],
}

# ============================================================================
# CRITERIOS DE DECISIÓN INTELIGENTE
# ============================================================================
SMART_DECISION_PARAMS = {
    'critical_node_bonus': 1.3,      # Bonus para nodos críticos
    'high_demand_threshold': 70,     # kVA para considerar alta demanda
    'high_risk_penalty': 0.6,        # Penaliza nodos con riesgo > 0.25
    'fast_install_bonus': 1.2,       # Bonus para instalación < 5 días
    'max_distance_penalty': 1000,    # Penaliza distancias > 1000m
}