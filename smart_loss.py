"""
smart_loss.py
=============
Loss function que penaliza basándose en un score ponderado,
no solo en la distancia pura.
"""

import torch
import torch.nn as nn
import config


class SmartMSTLoss(nn.Module):
    """
    Loss que considera múltiples factores más allá de la distancia.
    
    Formula del score ponderado:
    score = (distance * w_dist) + 
            (criticality_penalty * w_crit) + 
            (demand_factor * w_demand) +
            (risk_factor * w_risk) +
            (cost_factor * w_cost) +
            (time_factor * w_time)
    """
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Balance entre loss de clasificación y loss de score (0-1)
        """
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
        
        # Pesos desde config
        self.w_distance = config.FEATURE_WEIGHTS['distance']
        self.w_criticality = config.FEATURE_WEIGHTS['criticality']
        self.w_demand = config.FEATURE_WEIGHTS['demand']
        self.w_risk = config.FEATURE_WEIGHTS['risk']
        self.w_cost = config.FEATURE_WEIGHTS['cost']
        self.w_time = config.FEATURE_WEIGHTS['installation_time']
    
    def compute_edge_scores(self, data, normalize=True):
        """
        Calcula scores ponderados dinámicamente para cada arista,
        adaptándose a las columnas disponibles en data.x.
        """
        x = data.x  # [num_nodes, num_features]
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # ===============================
        # 1. DISTANCIA NORMALIZADA
        # ===============================
        distances = edge_attr.squeeze(-1)
        max_dist = distances.max() if distances.numel() > 0 else 1.0
        distance_score = distances / (max_dist + 1e-8)

        num_nodes, num_features = x.shape

        # ===============================
        # 2. CRITICIDAD (si existen esas columnas)
        # ===============================
        if num_features >= 15:
            criticidad_start = num_features - 4  # últimas 4 columnas
            criticidad_features = x[:, criticidad_start:criticidad_start+4]
        elif num_features >= 4:
            criticidad_start = num_features - 4
            criticidad_features = x[:, criticidad_start:criticidad_start+4]
        else:
            # Si no existen columnas de criticidad
            criticidad_features = torch.zeros((num_nodes, 4), device=x.device)

        criticidad_values = torch.tensor([0.8, 0.2, 1.0, 0.5], device=x.device)
        node_criticidad = (criticidad_features * criticidad_values).sum(dim=1)
        criticidad_score = (node_criticidad[src_nodes] + node_criticidad[dst_nodes]) / 2.0
        criticidad_penalty = 1.0 - criticidad_score

        # ===============================
        # 3. DEMANDA (columna 2 si existe)
        # ===============================
        if num_features > 2:
            demand_src = x[src_nodes, 2]
            demand_dst = x[dst_nodes, 2]
            demand_score = (demand_src + demand_dst) / 2.0
        else:
            demand_score = torch.zeros_like(distance_score)

        # ===============================
        # 4. RIESGO (columna 5 si existe)
        # ===============================
        if num_features > 5:
            risk_src = x[src_nodes, 5]
            risk_dst = x[dst_nodes, 5]
            risk_score = (risk_src + risk_dst) / 2.0
        else:
            risk_score = torch.zeros_like(distance_score)

        # ===============================
        # 5. COSTO (columna 3 si existe)
        # ===============================
        if num_features > 3:
            cost_src = x[src_nodes, 3]
            cost_dst = x[dst_nodes, 3]
            cost_score = (cost_src + cost_dst) / 2.0
        else:
            cost_score = torch.zeros_like(distance_score)

        # ===============================
        # 6. TIEMPO (columna 4 si existe)
        # ===============================
        if num_features > 4:
            time_src = x[src_nodes, 4]
            time_dst = x[dst_nodes, 4]
            time_score = (time_src + time_dst) / 2.0
        else:
            time_score = torch.zeros_like(distance_score)

        # ===============================
        # COMBINAR TODO
        # ===============================
        total_score = (
            self.w_distance * distance_score +
            self.w_criticality * criticidad_penalty +
            self.w_demand * demand_score +
            self.w_risk * risk_score +
            self.w_cost * cost_score +
            self.w_time * time_score
        )

        # Normalizar
        if normalize and total_score.numel() > 0:
            min_score = total_score.min()
            max_score = total_score.max()
            total_score = (total_score - min_score) / (max_score - min_score + 1e-8)

        return total_score

    
    def forward(self, logits, data):
        """
        Loss híbrido: clasificación + ranking de scores.
        
        Args:
            logits: Predicciones del modelo [num_edges]
            data: PyG Data con y (labels), x, edge_index, edge_attr
        
        Returns:
            Loss total
        """
        # Loss 1: Clasificación estándar (MST vs no-MST)
        bce = self.bce_loss(logits, data.y)
        
        # Loss 2: Ranking loss - aristas con mejor score deben tener mayor probabilidad
        edge_scores = self.compute_edge_scores(data, normalize=True)
        probabilities = torch.sigmoid(logits)
        
        # Margin ranking loss: si score_i < score_j entonces prob_i > prob_j
        # Invertimos porque menor score = mejor arista
        inverted_scores = 1.0 - edge_scores
        
        # Correlación negativa entre probabilidades y scores invertidos
        ranking_loss = torch.nn.functional.mse_loss(probabilities, inverted_scores)
        
        # Loss total
        total_loss = (1 - self.alpha) * bce + self.alpha * ranking_loss
        
        return total_loss


class WeightedMSTLoss(nn.Module):
    """
    Alternativa: Loss que usa el score como peso en la pérdida.
    """
    
    def __init__(self):
        super().__init__()
        self.smart_loss = SmartMSTLoss()
    
    def forward(self, logits, data):
        edge_scores = self.smart_loss.compute_edge_scores(data, normalize=True)
        
        # Invertir scores: menor score = mayor peso en loss
        weights = 1.0 - edge_scores + 0.1  # +0.1 para evitar weights=0
        
        # BCE con pesos dinámicos
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, 
            data.y, 
            reduction='none'
        )
        
        weighted_bce = (bce * weights).mean()
        
        return weighted_bce


# ============================================================================
# FUNCIÓN AUXILIAR PARA DEBUGGING
# ============================================================================
def analyze_edge_importance(data, model=None):
    """
    Analiza qué aristas son más importantes según el score ponderado.
    """
    smart_loss = SmartMSTLoss()
    scores = smart_loss.compute_edge_scores(data, normalize=False)
    
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    print("\n" + "="*70)
    print("ANÁLISIS DE IMPORTANCIA DE ARISTAS")
    print("="*70)
    
    # Top 10 aristas más importantes (menor score = mejor)
    sorted_indices = scores_np.argsort()[:10]
    
    print("\n🏆 TOP 10 ARISTAS MÁS IMPORTANTES:")
    for i, idx in enumerate(sorted_indices, 1):
        u, v = edge_index[0, idx], edge_index[1, idx]
        dist = edge_attr[idx, 0]
        score = scores_np[idx]
        
        print(f"  {i}. Arista ({u}, {v}): distancia={dist:.2f}m, score={score:.4f}")
        
        if model is not None:
            # Mostrar predicción del modelo
            with torch.no_grad():
                logit = model(data)[idx]
                prob = torch.sigmoid(logit).item()
                print(f"      Prob GNN: {prob:.4f}")
    
    print("="*70)