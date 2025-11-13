"""
config.py - VERSIÓN MEJORADA PARA DATASET RICO
================================================
Configuración optimizada para aprovechar características adicionales
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

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# PARÁMETROS DEL GRAFO REAL (TIBASOSA)
# ============================================================================
GRAPH_CONFIG = {
    'num_nodes': 17,
    'num_edges': None,
    'coordinates_file': 'tibasosa_coordenadas_enhanced.csv',  # 🔥 NUEVO ARCHIVO
    'graph_file': 'tibasosa_graph_enhanced.csv',
    'use_complete_graph': True,
    'max_distance_meters': None,
    'use_proximity_graph': False,
    'k_neighbors': 5,
}

# ============================================================================
# PARÁMETROS DEL DATASET SINTÉTICO
# ============================================================================
DATASET_CONFIG = {
    'num_graphs': 1000,        # MÁS GRAFOS para aprender mejor
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
# PARÁMETROS DEL MODELO GNN (GAT) - MEJORADOS
# ============================================================================
MODEL_CONFIG = {
    'hidden_channels': 64,    # MÁS CAPACIDAD (64→128)
    'num_heads': 4,            # MÁS ATENCIÓN (4→6)
    'num_layers': 3,           # MÁS PROFUNDIDAD (3→4)
    'dropout': 0.3,            # MÁS REGULARIZACIÓN (0.2→0.3)
    'activation': 'leaky_relu',
    'negative_slope': 0.2,
}

# ============================================================================
# PARÁMETROS DE ENTRENAMIENTO - OPTIMIZADOS
# ============================================================================
TRAINING_CONFIG = {
    'learning_rate': 0.0005,   # MÁS FINO (0.001→0.0005)
    'weight_decay': 1e-4,      # MÁS REGULARIZACIÓN
    'batch_size': 32,         # MÁS PEQUEÑO para mejor generalización
    'num_epochs': 200,         # MÁS ÉPOCAS (200→300)
    'early_stopping_patience': 40,  # MÁS PACIENCIA (20→40)
    'save_best_model': True,
    'log_interval': 10,
}

# ============================================================================
# PARÁMETROS DE EVALUACIÓN
# ============================================================================
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'optimality_gap'],
    'compare_algorithms': ['kruskal', 'prim', 'gnn_greedy'],
}

# ============================================================================
# PARÁMETROS DE VISUALIZACIÓN
# ============================================================================
VISUALIZATION_CONFIG = {
    'figure_size': (14, 10),   # MÁS GRANDE para ver detalles
    'node_size': 600,          # NODOS MÁS GRANDES
    'node_color': '#1f77b4',
    'edge_width': 2.5,         # ARISTAS MÁS GRUESAS
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
# PESOS PARA DECISIÓN DEL GNN (NUEVA SECCIÓN)
# ============================================================================
# Estos pesos influencian cómo el modelo pondera diferentes características
FEATURE_WEIGHTS = {
    # PRINCIPALES (obligatorias en mayoría de casos)
    'distance': 0.25,              # 25% - Distancia física
    'criticality': 0.15,           # 15% - Criticidad (critica > alta > media > baja)
    'demand': 0.12,                # 12% - Demanda proyectada
    'risk': 0.10,                  # 10% - Factor de riesgo (invertido)
    'cost': 0.08,                  # 8% - Costo de mantenimiento
    
    # SECUNDARIAS (útiles si disponibles)
    'installation_time': 0.06,     # 6% - Tiempo de instalación
    'capacity': 0.08,              # 8% - Capacidad kVA
    'priority': 0.08,              # 8% - Prioridad (alta > media > baja)
    
    # CONTEXTUALES (opcionales)
    'zone': 0.03,                  # 3% - Zona (comercial/urbana mejor)
    'accessibility': 0.03,         # 3% - Accesibilidad
    'land_availability': 0.02,     # 2% - Disponibilidad de terreno
}

# Verificación automática
_total_weight = sum(FEATURE_WEIGHTS.values())
if abs(_total_weight - 1.0) > 0.01:
    print(f"⚠️ ADVERTENCIA: Suma de FEATURE_WEIGHTS = {_total_weight:.3f} (debería ser ~1.0)")
else:
    print(f"✅ FEATURE_WEIGHTS correctamente balanceado: {_total_weight:.3f}")

# ============================================================================
# CRITERIOS DE DECISIÓN INTELIGENTE
# ============================================================================
SMART_DECISION_PARAMS = {
    'critical_node_bonus': 1.2,      # Multiplica por 1.2 el score de nodos críticos
    'high_demand_threshold': 70,     # kVA para considerar alta demanda
    'high_risk_penalty': 0.7,        # Penaliza x0.7 nodos con riesgo > 0.25
    'fast_install_bonus': 1.15,      # Bonus para instalación < 5 días
    'max_distance_penalty': 1000,    # Penaliza fuertemente distancias > 1000m
}