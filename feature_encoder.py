"""
feature_encoder.py
==================
Codificador automático de características para grafos.
Detecta y normaliza columnas numéricas y categóricas automáticamente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch


class AutoFeatureEncoder:
    """
    Codificador que automáticamente detecta tipos de datos y los normaliza.
    - Columnas numéricas: normalización min-max
    - Columnas categóricas: one-hot encoding
    - Ignora columnas especiales: nombre, lat, lon, node_id
    """
    
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_ranges = {}  # {col: (min, max)}
        self.categorical_mappings = {}  # {col: {value: index}}
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'AutoFeatureEncoder':
        """
        Analiza el DataFrame y aprende las transformaciones necesarias.
        
        Args:
            df: DataFrame con todas las columnas
        
        Returns:
            self (para encadenamiento)
        """
        # Columnas a ignorar
        ignore_cols = ['nombre', 'name', 'node_id', 'lat', 'latitud', 
                       'lon', 'longitud', 'pos', 'x_coord', 'y_coord']
        
        # Detectar columnas numéricas y categóricas
        for col in df.columns:
            col_lower = col.lower()
            
            # Ignorar columnas especiales
            if any(ig in col_lower for ig in ignore_cols):
                continue
            
            # Verificar si es numérica
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
                # Guardar rango para normalización
                self.numeric_ranges[col] = (df[col].min(), df[col].max())
            else:
                # Es categórica
                self.categorical_cols.append(col)
                # Crear mapeo de categorías
                unique_values = df[col].dropna().unique()
                self.categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(sorted(unique_values))
                }
        
        # Construir nombres de características finales
        self.feature_names = []
        
        # Características numéricas normalizadas
        for col in self.numeric_cols:
            self.feature_names.append(f"{col}_norm")
        
        # Características one-hot de categóricas
        for col in self.categorical_cols:
            for val in sorted(self.categorical_mappings[col].keys()):
                self.feature_names.append(f"{col}_{val}")
        
        self.is_fitted = True
        

        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma el DataFrame en matriz de características normalizadas.
        
        Args:
            df: DataFrame con los datos
        
        Returns:
            Matriz numpy de forma (num_nodes, num_features)
        """
        if not self.is_fitted:
            raise ValueError("Encoder no entrenado. Llama primero a fit()")
        
        features_list = []
        
        # 1. Normalizar columnas numéricas
        for col in self.numeric_cols:
            min_val, max_val = self.numeric_ranges[col]
            
            # Normalización min-max con manejo de constantes
            if max_val == min_val:
                normalized = np.ones(len(df)) * 0.5
            else:
                normalized = (df[col] - min_val) / (max_val - min_val)
            
            features_list.append(normalized.values.reshape(-1, 1))
        
        # 2. One-hot encoding de categóricas
        for col in self.categorical_cols:
            mapping = self.categorical_mappings[col]
            num_categories = len(mapping)
            
            # Crear matriz one-hot
            one_hot = np.zeros((len(df), num_categories))
            
            for idx, value in enumerate(df[col]):
                if pd.notna(value) and value in mapping:
                    category_idx = mapping[value]
                    one_hot[idx, category_idx] = 1.0
            
            features_list.append(one_hot)
        
        # Concatenar todas las características
        if features_list:
            features = np.hstack(features_list)
        else:
            # Si no hay características, usar vector de unos
            features = np.ones((len(df), 1))
        
        return features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Entrena y transforma en un solo paso.
        
        Args:
            df: DataFrame con los datos
        
        Returns:
            Matriz de características
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nombres de características generadas."""
        return self.feature_names.copy()
    
    def get_num_features(self) -> int:
        """Retorna número total de características."""
        return len(self.feature_names)
    
    def summary(self) -> Dict:
        """
        Retorna resumen de la codificación.
        
        Returns:
            Diccionario con información del encoder
        """
        return {
            'num_numeric_cols': len(self.numeric_cols),
            'num_categorical_cols': len(self.categorical_cols),
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'categorical_mappings': self.categorical_mappings
        }


def encode_node_features(df: pd.DataFrame, 
                        degree_dict: Dict[int, int] = None,
                        num_nodes: int = None) -> Tuple[np.ndarray, AutoFeatureEncoder]:
    """
    Función auxiliar para codificar características de nodos de forma automática.
    
    Args:
        df: DataFrame con información de nodos
        degree_dict: Diccionario {node_id: degree} opcional
        num_nodes: Número total de nodos (para normalizar grado)
    
    Returns:
        Tupla (matriz_features, encoder)
    """
    encoder = AutoFeatureEncoder()
    
    # Codificar características del DataFrame
    features = encoder.fit_transform(df)
    
    # Agregar grado normalizado si se proporciona
    if degree_dict is not None and num_nodes is not None:
        degree_features = np.array([
            [degree_dict.get(i+1, 0) / num_nodes] 
            for i in range(len(df))
        ])
        features = np.hstack([degree_features, features])
        encoder.feature_names.insert(0, 'degree_norm')
    
    return features, encoder


# ============================================================================
# EJEMPLO DE USO
# ============================================================================
if __name__ == "__main__":
    # Crear DataFrame de ejemplo
    data = {
        'Nombre': ['Hotel A', 'Comercio B', 'Publico C'],
        'latitud': [5.75, 5.74, 5.73],
        'longitud': [-73.00, -73.01, -73.02],
        'tipo': ['residencial', 'comercial', 'publico'],
        'capacidad_kva': [45, 75, 30],
        'prioridad': ['media', 'alta', 'alta']
    }
    df = pd.DataFrame(data)
    
    print("DataFrame original:")
    print(df)
    print()
    
    # Codificar automáticamente
    encoder = AutoFeatureEncoder()
    features = encoder.fit_transform(df)
    
    print("\nMatriz de características:")
    print(features)
    print(f"Shape: {features.shape}")
    
    print("\nResumen:")
    summary = encoder.summary()
    for key, value in summary.items():
        print(f"{key}: {value}")