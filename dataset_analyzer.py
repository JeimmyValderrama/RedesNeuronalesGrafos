"""
dataset_analyzer.py
===================
Analiza automáticamente el dataset objetivo y extrae sus características
para generar datos sintéticos compatibles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
import config


class DatasetAnalyzer:
    """
    Analiza un dataset de entrada y extrae metadatos para replicar
    su estructura en datos sintéticos.
    """
    
    def __init__(self):
        self.schema = {
            'num_features': 0,
            'feature_names': [],
            'numeric_features': [],
            'categorical_features': [],
            'feature_ranges': {},
            'categorical_distributions': {},
            'has_degree': False
        }
        self.analyzed = False
    
    def analyze_from_csv(self, csv_path: Path) -> Dict:
        """
        Analiza un archivo CSV y extrae su esquema de características.
        
        Args:
            csv_path: Ruta al archivo CSV con datos
        
        Returns:
            Diccionario con esquema de características
        """
        print("\n" + "=" * 70)
        print("ANALIZANDO ESTRUCTURA DEL DATASET OBJETIVO")
        print("=" * 70)
        
        # Leer CSV
        try:
            df = pd.read_csv(csv_path, encoding='latin1', sep=';', skip_blank_lines=True)
        except:
            df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Normalizar nombres de columnas
        df.columns = df.columns.str.strip().str.lower()
        
        print(f"\n✓ Archivo cargado: {csv_path.name}")
        print(f"  Filas: {len(df)}")
        print(f"  Columnas: {list(df.columns)}")
        
        # Identificar columnas especiales (ignorar en características)
        ignore_cols = ['nombre', 'name', 'node_id', 'lat', 'latitud', 
                       'lon', 'longitud', 'pos', 'x_coord', 'y_coord']
        
        feature_cols = [col for col in df.columns 
                       if not any(ig in col.lower() for ig in ignore_cols)]
        
        print(f"\n📊 Columnas de características detectadas: {len(feature_cols)}")
        
        # Analizar cada característica
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Característica numérica
                self.schema['numeric_features'].append(col)
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                self.schema['feature_ranges'][col] = {
                    'min': float(min_val),
                    'max': float(max_val),
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'type': 'numeric'
                }
                
                print(f"  • {col} (numérica): [{min_val:.2f}, {max_val:.2f}], μ={mean_val:.2f}")
            
            else:
                # Característica categórica
                self.schema['categorical_features'].append(col)
                
                # Calcular distribución
                value_counts = df[col].value_counts(normalize=True)
                categories = value_counts.index.tolist()
                probabilities = value_counts.values.tolist()
                
                self.schema['categorical_distributions'][col] = {
                    'categories': categories,
                    'probabilities': probabilities,
                    'type': 'categorical'
                }
                
                print(f"  • {col} (categórica): {len(categories)} categorías")
                for cat, prob in zip(categories, probabilities):
                    print(f"      - {cat}: {prob*100:.1f}%")
        
        # Calcular número total de características después de encoding
        num_features = len(self.schema['numeric_features'])
        
        for col in self.schema['categorical_features']:
            num_categories = len(self.schema['categorical_distributions'][col]['categories'])
            num_features += num_categories  # One-hot encoding
        
        # Agregar 1 para el grado normalizado (siempre presente)
        num_features += 1
        self.schema['has_degree'] = True
        
        self.schema['num_features'] = num_features
        
        # Construir nombres de características finales
        feature_names = ['degree_norm']
        
        for col in self.schema['numeric_features']:
            feature_names.append(f"{col}_norm")
        
        for col in self.schema['categorical_features']:
            for cat in self.schema['categorical_distributions'][col]['categories']:
                feature_names.append(f"{col}_{cat}")
        
        self.schema['feature_names'] = feature_names
    
        for i, name in enumerate(feature_names, 1):
            print(f"  {i}. {name}")
        
        print("=" * 70)
        
        self.analyzed = True
        return self.schema
    
    def generate_synthetic_features(self, num_nodes: int, degrees: Dict[int, int]) -> np.ndarray:
        """
        Genera características sintéticas basadas en el esquema analizado.
        
        Args:
            num_nodes: Número de nodos
            degrees: Diccionario {node_id: degree}
        
        Returns:
            Matriz de características numpy (num_nodes, num_features)
        """
        if not self.analyzed:
            raise ValueError("Primero debes analizar un dataset con analyze_from_csv()")
        
        features_list = []
        
        # 1. Grado normalizado
        degree_features = np.array([[degrees.get(i, 0) / num_nodes] for i in range(num_nodes)])
        features_list.append(degree_features)
        
        # 2. Características numéricas
        for col in self.schema['numeric_features']:
            info = self.schema['feature_ranges'][col]
            
            # Generar valores siguiendo distribución normal truncada
            mean = info['mean']
            std = info['std']
            min_val = info['min']
            max_val = info['max']
            
            # Generar con distribución normal y truncar
            values = np.random.normal(mean, std, num_nodes)
            values = np.clip(values, min_val, max_val)
            
            # Normalizar a [0, 1]
            if max_val > min_val:
                values_norm = (values - min_val) / (max_val - min_val)
            else:
                values_norm = np.ones(num_nodes) * 0.5
            
            features_list.append(values_norm.reshape(-1, 1))
        
        # 3. Características categóricas (one-hot)
        for col in self.schema['categorical_features']:
            dist = self.schema['categorical_distributions'][col]
            categories = dist['categories']
            probabilities = dist['probabilities']
            num_categories = len(categories)
            
            # Generar categorías para cada nodo según la distribución
            chosen_indices = np.random.choice(
                len(categories), 
                size=num_nodes, 
                p=probabilities
            )
            
            # Crear one-hot encoding
            one_hot = np.zeros((num_nodes, num_categories))
            one_hot[np.arange(num_nodes), chosen_indices] = 1.0
            
            features_list.append(one_hot)
        
        # Concatenar todas las características
        features = np.hstack(features_list)
        
        return features
    
    def save_schema(self, file_path: Path):
        """Guarda el esquema analizado en JSON."""
        with open(file_path, 'w') as f:
            json.dump(self.schema, f, indent=2)
        print(f"✓ Esquema guardado en: {file_path}")
    
    def load_schema(self, file_path: Path):
        """Carga un esquema previamente guardado."""
        with open(file_path, 'r') as f:
            self.schema = json.load(f)
        self.analyzed = True
        print(f"✓ Esquema cargado desde: {file_path}")
        return self.schema
    
    def get_num_features(self) -> int:
        """Retorna el número total de características."""
        return self.schema['num_features']
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nombres de características."""
        return self.schema['feature_names']
    
    def summary(self) -> Dict:
        """Retorna resumen completo del esquema."""
        return self.schema.copy()


def analyze_target_dataset(csv_file: str = None) -> DatasetAnalyzer:
    """
    Función auxiliar para analizar rápidamente un dataset objetivo.
    
    Args:
        csv_file: Nombre del archivo CSV (busca en config.DATA_DIR)
    
    Returns:
        Analizador configurado
    """
    if csv_file is None:
        csv_file = config.GRAPH_CONFIG.get('coordinates_file', 'tibasosa_coordenadas.csv')
    
    csv_path = config.DATA_DIR / csv_file
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
    
    analyzer = DatasetAnalyzer()
    analyzer.analyze_from_csv(csv_path)
    
    # Guardar esquema para uso futuro
    schema_path = config.DATA_DIR / 'dataset_schema.json'
    analyzer.save_schema(schema_path)
    
    return analyzer


