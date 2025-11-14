"""
smart_loss.py - VERSIÓN 100% DINÁMICA Y COMPLETA
=================================================
Usa TODAS las características del dataset automáticamente.
"""

import torch
import torch.nn as nn
import config
from pathlib import Path
import json


class SmartMSTLoss(nn.Module):
    """
    Loss que considera TODAS las características de forma COMPLETAMENTE DINÁMICA.
    """
    
    def __init__(self, alpha=config.ALPHA, schema_path: Path = None):
        """
        Args:
            alpha: Balance entre BCE y ranking (0-1). Más alto = más peso a características
        """
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
        
        # Cargar schema
        self.schema = self._load_schema(schema_path)
        
        if self.schema:
            print(f" SmartMSTLoss inicializado (alpha={alpha})")

            num_features = self.schema.get('num_features', 0)
            print(f"   • Schema cargado: {num_features} características")
        else:
            print(f" SmartMSTLoss sin schema, modo básico")
    
    def _load_schema(self, schema_path: Path = None) -> dict:
        """Carga el schema del dataset."""
        if schema_path is None:
            schema_path = config.DATA_DIR / 'dataset_schema.json'
        
        if not schema_path.exists():
            return None
        
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _get_feature_index(self, feature_name: str):
        """Obtiene índice de característica de forma flexible."""
        if not self.schema or 'feature_indices' not in self.schema:
            return None
        
        indices = self.schema['feature_indices']
        
        # Buscar coincidencias
        for key, value in indices.items():
            if feature_name.lower() in key.lower() or key.lower() in feature_name.lower():
                return value
        
        return None
    
    def _get_categorical_values(self, feature_key: str) -> torch.Tensor:
        """Obtiene valores categóricos basándose en el feature_key."""
        if not self.schema or 'categorical_distributions' not in self.schema:
            return None
        
        # Buscar la distribución
        for key, dist in self.schema['categorical_distributions'].items():
            if feature_key.lower() in key.lower():
                categories = dist['categories']
                
                # Mapeo de valores según tipo
                if 'tipo' in key.lower() or 'type' in key.lower():
                    value_map = {'residencial': 0.5, 'comercial': 0.7, 'publico': 0.9, 'público': 0.9}
                elif 'prioridad' in key.lower() or 'priority' in key.lower():
                    value_map = {'baja': 0.3, 'media': 0.6, 'alta': 1.0}
                elif 'criticidad' in key.lower() or 'critica' in key.lower():
                    value_map = {'baja': 0.2, 'media': 0.5, 'alta': 0.8, 'critica': 1.0}
                elif 'zona' in key.lower() or 'zone' in key.lower():
                    value_map = {'rural': 0.3, 'urbana': 0.8, 'comercial': 0.9, 'recreativa': 0.6, 'educativa': 0.7}
                elif 'accesibilidad' in key.lower() or 'access' in key.lower():
                    value_map = {'dificil': 0.3, 'difícil': 0.3, 'media': 0.6, 'facil': 1.0, 'fácil': 1.0}
                elif 'disponibilidad' in key.lower() or 'availability' in key.lower():
                    value_map = {'baja': 0.3, 'media': 0.6, 'alta': 1.0}
                else:
                    return torch.linspace(0.2, 1.0, len(categories))
                
                values = [value_map.get(cat.lower(), 0.5) for cat in categories]
                return torch.tensor(values)
        
        return None
    
    def compute_edge_scores(self, data, normalize=True):
        """
        Calcula scores usando TODAS las características disponibles dinámicamente.
        CORREGIDO: Mejor normalización y lógica de scores.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        num_nodes, num_features = x.shape

        # 1. DISTANCIA (INVERTIDA - menor distancia = mejor)
        distances = edge_attr.squeeze(-1)
        max_dist = distances.max() if distances.numel() > 0 else 1.0
        distance_score = 1.0 - (distances / (max_dist + 1e-8))  # ¡CORREGIDO!
        
        weight_distance = config.FEATURE_WEIGHTS.get('distance', 0.15)
        total_score = weight_distance * distance_score
        
        if not self.schema:
            if normalize:
                min_score = total_score.min()
                max_score = total_score.max()
                total_score = (total_score - min_score) / (max_score - min_score + 1e-8)
            return total_score

        # 2. CARACTERÍSTICAS NUMÉRICAS (NORMALIZADAS)
        numeric_features = [
            ('capacity', ['capacidad', 'capacidad_kva'], False),  # (key, terms, invert)
            ('demand', ['demanda', 'demanda_proyectada'], False),
            ('cost', ['costo', 'costo_mantenimiento'], True),     # ¡Menor costo = mejor!
            ('installation_time', ['tiempo', 'tiempo_instalacion'], True),  # ¡Menor tiempo = mejor!
            ('risk', ['riesgo', 'factor_riesgo'], True),          # ¡Menor riesgo = mejor!
        ]
        
        for feature_key, search_terms, should_invert in numeric_features:
            weight = config.FEATURE_WEIGHTS.get(feature_key, 0.0)
            
            if weight == 0:
                continue
            
            # Buscar índice
            idx = None
            for term in search_terms:
                idx = self._get_feature_index(term)
                if idx is not None:
                    break
            
            if idx is not None and isinstance(idx, int) and idx < num_features:
                feat_src = x[src_nodes, idx]
                feat_dst = x[dst_nodes, idx]
                
                # NORMALIZAR: llevar a rango [0,1]
                feature_max = x[:, idx].max()
                feature_min = x[:, idx].min()
                
                if feature_max > feature_min:
                    feat_src_norm = (feat_src - feature_min) / (feature_max - feature_min)
                    feat_dst_norm = (feat_dst - feature_min) / (feature_max - feature_min)
                    feat_score = (feat_src_norm + feat_dst_norm) / 2.0
                else:
                    feat_score = (feat_src + feat_dst) / 2.0
                
                # INVERTIR si es necesario (menor = mejor)
                if should_invert:
                    feat_score = 1.0 - feat_score
                
                total_score += weight * feat_score

        # 3. CARACTERÍSTICAS CATEGÓRICAS (ya están en [0,1])
        categorical_features = [
            ('type', ['tipo']),
            ('priority', ['prioridad']),
            ('criticality', ['criticidad']),
            ('zone', ['zona']),
            ('accessibility', ['accesibilidad']),
            ('land_availability', ['disponibilidad', 'disponibilidad_terreno']),
        ]
        
        for feature_key, search_terms in categorical_features:
            weight = config.FEATURE_WEIGHTS.get(feature_key, 0.0)
            
            if weight == 0:
                continue
            
            # Buscar índice
            idx_range = None
            for term in search_terms:
                idx_range = self._get_feature_index(term)
                if idx_range is not None:
                    break
            
            if idx_range is not None and isinstance(idx_range, tuple):
                start_idx, end_idx = idx_range
                
                if end_idx <= num_features:
                    cat_values = self._get_categorical_values(feature_key)
                    
                    if cat_values is not None:
                        cat_values = cat_values.to(x.device)
                        cat_features = x[:, start_idx:end_idx]
                        
                        cat_score_nodes = (cat_features * cat_values).sum(dim=1)
                        cat_score_edge = (cat_score_nodes[src_nodes] + cat_score_nodes[dst_nodes]) / 2.0
                        
                        total_score += weight * cat_score_edge

        # Normalizar
        if normalize and total_score.numel() > 0:
            min_score = total_score.min()
            max_score = total_score.max()
            if max_score > min_score:
                total_score = (total_score - min_score) / (max_score - min_score + 1e-8)

        return total_score
    
    def forward(self, logits, data):
        """Loss híbrido: clasificación + ranking."""
        # Loss 1: Clasificación
        bce = self.bce_loss(logits, data.y)
        
        # Loss 2: Ranking
        edge_scores = self.compute_edge_scores(data, normalize=True)
        probabilities = torch.sigmoid(logits)
        
        inverted_scores = 1.0 - edge_scores
        ranking_loss = torch.nn.functional.mse_loss(probabilities, inverted_scores)
        
        # Loss total (alpha más alto = más peso a características)
        total_loss = (1 - self.alpha) * bce + self.alpha * ranking_loss
        
        return total_loss


def analyze_edge_importance(data, model=None):
    """Analiza qué aristas son más importantes."""
    smart_loss = SmartMSTLoss()
    scores = smart_loss.compute_edge_scores(data, normalize=False)
    
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    print("\n" + "="*70)
    print("ANÁLISIS DE IMPORTANCIA DE ARISTAS (100% Dinámico)")
    print("="*70)
    
    sorted_indices = scores_np.argsort()[:10]
    
    print("\n TOP 10 ARISTAS MÁS IMPORTANTES:")
    for i, idx in enumerate(sorted_indices, 1):
        u, v = edge_index[0, idx], edge_index[1, idx]
        dist = edge_attr[idx, 0]
        score = scores_np[idx]
        
        print(f"  {i}. Arista ({u}, {v}): distancia={dist:.2f}m, score={score:.4f}")
        
        if model is not None:
            with torch.no_grad():
                logit = model(data)[idx]
                prob = torch.sigmoid(logit).item()
                print(f"      Prob GNN: {prob:.4f}")
    
    print("="*70)