"""
mst_algorithms_smart.py - VERSIÓN CORREGIDA
=============================================
Algoritmo GNN que USA las características para tomar decisiones diferentes.
"""

import networkx as nx
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import numpy as np
from model_gat import EdgeGAT
import config


class MSTSolver:
    """Solver con algoritmo inteligente que REALMENTE usa características."""
    
    def __init__(self, G: nx.Graph):
        self.G = G
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()
    
    def kruskal(self) -> Tuple[nx.Graph, float]:
        """MST usando Kruskal (solo distancia)."""
        mst = nx.minimum_spanning_tree(self.G, weight='weight', algorithm='kruskal')
        total_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
        return mst, total_weight
    
    def prim(self) -> Tuple[nx.Graph, float]:
        """MST usando Prim (solo distancia)."""
        mst = nx.minimum_spanning_tree(self.G, weight='weight', algorithm='prim')
        total_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
        return mst, total_weight
    
    def gnn_smart_greedy(self, 
                        model: EdgeGAT, 
                        data: Data,
                        device: torch.device = None,
                        use_smart_scoring: bool = True) -> Tuple[nx.Graph, float, Dict]:
        """
        Algoritmo INTELIGENTE que combina GNN + características del grafo.
        
        Args:
            model: Modelo GAT entrenado
            data: Datos PyG con características
            device: Dispositivo
            use_smart_scoring: Si True, usa scoring multifactorial
        
        Returns:
            Tupla (MST, peso_total, metadatos_decisiones)
        """
        device = device or config.DEVICE
        model = model.to(device)
        model.eval()
        
        # 1. PREDECIR PROBABILIDADES GNN
        with torch.no_grad():
            data = data.to(device)
            probabilities = model.predict_probabilities(data)
        
        # 2. EXTRAER DATOS
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy().flatten()
        probs = probabilities.cpu().numpy()
        node_features = data.x.cpu().numpy()
        
        # 3. CONSTRUIR LISTA DE ARISTAS CON SCORING INTELIGENTE
        edges_with_scores = []
        seen_edges = set()
        
        decision_log = {
            'edges_evaluated': 0,
            'smart_choices': 0,  # Aristas elegidas por características
            'distance_choices': 0,  # Aristas elegidas por distancia
            'feature_importance': {}
        }
        
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            if u < v:
                edge_tuple = (u, v)
                if edge_tuple not in seen_edges:
                    weight = edge_attr[i]
                    prob = probs[i]
                    
                    if use_smart_scoring:
                        # 🔥 CALCULAR SCORE INTELIGENTE
                        smart_score = self._compute_smart_score(
                            u, v, weight, prob, node_features
                        )
                        edges_with_scores.append((smart_score, weight, prob, u, v))
                    else:
                        # Solo probabilidad (comportamiento original)
                        edges_with_scores.append((prob, weight, prob, u, v))
                    
                    seen_edges.add(edge_tuple)
                    decision_log['edges_evaluated'] += 1
        
        # 4. ORDENAR POR SCORE (descendente = mejor score primero)
        edges_with_scores.sort(reverse=True, key=lambda x: x[0])
        
        # 5. ALGORITMO GREEDY CON UNION-FIND
        parent = list(range(self.num_nodes))
        rank = [0] * self.num_nodes
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        # 6. CONSTRUIR MST
        mst_edges = []
        total_weight = 0
        
        for smart_score, weight, prob, u, v in edges_with_scores:
            if union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight
                
                # Log de decisión
                if smart_score != prob:  # Se usó scoring inteligente
                    decision_log['smart_choices'] += 1
                else:
                    decision_log['distance_choices'] += 1
                
                if len(mst_edges) == self.num_nodes - 1:
                    break
        
        # 7. CONSTRUIR GRAFO MST
        mst = nx.Graph()
        for u, v, w in mst_edges:
            mst.add_edge(u, v, weight=w)
        
        return mst, total_weight, decision_log
    
    def _compute_smart_score(self, u: int, v: int, distance: float, 
                            prob_gnn: float, node_features: np.ndarray) -> float:
        """
        Calcula score MULTIFACTORIAL que considera:
        - Probabilidad GNN (30%)
        - Distancia normalizada (30%)
        - Características de los nodos (40%)
        """
        num_nodes, num_features = node_features.shape
        
        # COMPONENTE 1: Probabilidad GNN (ya entrenada con características)
        score_gnn = prob_gnn * 0.30
        
        # COMPONENTE 2: Distancia (invertida, menor es mejor)
        max_dist = 2000.0  # Máxima distancia esperada en metros
        score_distance = (1.0 - min(distance / max_dist, 1.0)) * 0.30
        
        # COMPONENTE 3: Características de los nodos (40%)
        feat_u = node_features[u]
        feat_v = node_features[v]
        
        # Detectar qué características existen basándose en num_features
        score_features = 0.0
        
        # CRITICIDAD (buscar en características categóricas al final)
        if num_features >= 15:  # Dataset con criticidad
            # Las últimas 4 features suelen ser criticidad
            criticidad_start = num_features - 11  # Ajustar según estructura
            criticidad_features_u = feat_u[criticidad_start:criticidad_start+4]
            criticidad_features_v = feat_v[criticidad_start:criticidad_start+4]
            
            # Valores: [baja, media, alta, critica]
            criticidad_weights = np.array([0.2, 0.4, 0.7, 1.0])
            criticidad_u = np.dot(criticidad_features_u, criticidad_weights)
            criticidad_v = np.dot(criticidad_features_v, criticidad_weights)
            criticidad_score = (criticidad_u + criticidad_v) / 2.0
            
            score_features += criticidad_score * 0.35  # 35% de las features
        
        # DEMANDA PROYECTADA (feature 2 si existe)
        if num_features > 2:
            demanda_u = feat_u[2]
            demanda_v = feat_v[2]
            demanda_score = (demanda_u + demanda_v) / 2.0
            score_features += demanda_score * 0.25  # 25% de las features
        
        # FACTOR DE RIESGO (feature 5 si existe) - invertido (menos riesgo = mejor)
        if num_features > 5:
            riesgo_u = feat_u[5]
            riesgo_v = feat_v[5]
            riesgo_score = 1.0 - ((riesgo_u + riesgo_v) / 2.0)
            score_features += riesgo_score * 0.20  # 20% de las features
        
        # CAPACIDAD (feature 1 si existe)
        if num_features > 1:
            capacidad_u = feat_u[1]
            capacidad_v = feat_v[1]
            capacidad_score = (capacidad_u + capacidad_v) / 2.0
            score_features += capacidad_score * 0.20  # 20% de las features
        
        # Normalizar score_features
        if num_features > 1:
            score_features *= 0.40  # 40% del score total
        
        # SCORE FINAL
        total_score = score_gnn + score_distance + score_features
        
        return total_score


def compare_mst_algorithms(G: nx.Graph, 
                           model: EdgeGAT = None,
                           data: Data = None) -> Dict:
    """
    Compara algoritmos incluyendo GNN INTELIGENTE.
    """
    solver = MSTSolver(G)
    results = {}
    
    # Kruskal
    print("Ejecutando Kruskal (solo distancia)...")
    mst_kruskal, weight_kruskal = solver.kruskal()
    results['kruskal'] = {
        'mst': mst_kruskal,
        'weight': weight_kruskal,
        'num_edges': mst_kruskal.number_of_edges(),
        'method': 'Greedy por distancia'
    }
    
    # Prim
    print("Ejecutando Prim (solo distancia)...")
    mst_prim, weight_prim = solver.prim()
    results['prim'] = {
        'mst': mst_prim,
        'weight': weight_prim,
        'num_edges': mst_prim.number_of_edges(),
        'method': 'Greedy por distancia'
    }
    
 
    if model is not None and data is not None:
        # GNN INTELIGENTE (score multifactorial)
        print("Ejecutando GNN(scoring multifactorial)...")
        mst_gnn_smart, weight_gnn_smart, log_smart = solver.gnn_smart_greedy(
            model, data, use_smart_scoring=True
        )
        results['gnn_smart'] = {
            'mst': mst_gnn_smart,
            'weight': weight_gnn_smart,
            'num_edges': mst_gnn_smart.number_of_edges(),
            'method': 'GNN + Características',
            'decision_log': log_smart
        }
        
        gap_smart = ((weight_gnn_smart - weight_kruskal) / weight_kruskal) * 100
        results['gnn_smart']['optimality_gap'] = gap_smart
    
    return results


def print_comparison_results(results: Dict):
    """Imprime resultados con análisis de decisiones."""
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE ALGORITMOS MST")
    print("=" * 80)
    
    for algo_name, result in results.items():
        print(f"\n{'🔷' if 'gnn' in algo_name else '🔹'} {algo_name.upper()}:")
        print(f"  • Método: {result.get('method', 'Clásico')}")
        print(f"  • Peso total: {result['weight']:.2f} metros")
        print(f"  • Aristas en MST: {result['num_edges']}")
        
        if 'decision_log' in result:
            log = result['decision_log']
            print(f"  • Decisiones inteligentes: {log['smart_choices']}/{log['edges_evaluated']}")
            print(f"  • Decisiones por distancia: {log['distance_choices']}/{log['edges_evaluated']}")
        
        if 'optimality_gap' in result:
            gap = result['optimality_gap']
            print(f"  • Gap vs óptimo: {gap:+.2f}%")
            
            if abs(gap) < 0.01:
                print(f"  ✅ ¡Solución ÓPTIMA!")
            elif gap < 0:
                print(f"  🏆 ¡MEJOR que Kruskal! (imposible, revisar)")
            elif gap < 5:
                print(f"  ✅ Excelente (<5% extra)")
            elif gap < 10:
                print(f"  ⚠️ Aceptable (5-10% extra)")
            else:
                print(f"  📊 Sacrificio por criterios inteligentes (>{gap:.1f}%)")
    
    print("=" * 80)


def extract_mst_solution(mst: nx.Graph) -> List[Tuple[int, int, float]]:
    """Extrae lista de aristas del MST."""
    edges = []
    for u, v, data in mst.edges(data=True):
        edges.append((u, v, data['weight']))
    return edges


def validate_mst(mst: nx.Graph, original_graph: nx.Graph) -> Dict:
    """Valida que un MST sea correcto."""
    validation = {}
    
    validation['is_connected'] = nx.is_connected(mst)
    validation['is_tree'] = nx.is_tree(mst)
    
    n = original_graph.number_of_nodes()
    validation['has_correct_edges'] = (mst.number_of_edges() == n - 1)
    validation['has_all_nodes'] = (set(mst.nodes()) == set(original_graph.nodes()))
    
    all_edges_valid = True
    for u, v in mst.edges():
        if not original_graph.has_edge(u, v):
            all_edges_valid = False
            break
    validation['all_edges_valid'] = all_edges_valid
    
    validation['is_valid'] = all(validation.values())
    
    return validation


def analyze_decision_differences(G: nx.Graph, results: Dict, data: Data):
    """
    Analiza POR QUÉ GNN-Smart eligió aristas diferentes.
    """
    if 'gnn_smart' not in results or 'kruskal' not in results:
        return
    
    print("\n" + "=" * 80)
    print("ANÁLISIS DE DECISIONES DIFERENTES")
    print("=" * 80)
    
    mst_kruskal = results['kruskal']['mst']
    mst_gnn = results['gnn_smart']['mst']
    
    kruskal_edges = set(tuple(sorted([u, v])) for u, v in mst_kruskal.edges())
    gnn_edges = set(tuple(sorted([u, v])) for u, v in mst_gnn.edges())
    
    only_kruskal = kruskal_edges - gnn_edges
    only_gnn = gnn_edges - kruskal_edges
    
    if not only_gnn:
        print("\n⚠️ GNN eligió las MISMAS aristas que Kruskal")
        print("   Posibles causas:")
        print("   1. Pocas características diferentes entre nodos")
        print("   2. Modelo no entrenó suficiente con características")
        print("   3. Características no son discriminativas")
        return
    
    print(f"\n🔍 ARISTAS DIFERENTES:")
    print(f"   Solo en Kruskal: {len(only_kruskal)}")
    print(f"   Solo en GNN-Smart: {len(only_gnn)}")
    
    node_features = data.x.cpu().numpy()
    
    print(f"\n💡 ARISTAS QUE GNN ELIGIÓ (en lugar de Kruskal):")
    for i, (u, v) in enumerate(list(only_gnn)[:5], 1):
        dist_gnn = G[u][v]['weight']
        
        print(f"\n{i}. Arista ({u}, {v}): {dist_gnn:.2f}m")
        
        # Analizar características
        if 'name' in G.nodes[u]:
            print(f"   {G.nodes[u].get('name', f'Nodo {u}')} ↔ {G.nodes[v].get('name', f'Nodo {v}')}")
        
        # Mostrar características relevantes
        feat_u = node_features[u]
        feat_v = node_features[v]
        
        if len(feat_u) > 5:
            print(f"   Nodo {u}: capacidad={feat_u[1]:.2f}, demanda={feat_u[2]:.2f}, riesgo={feat_u[5]:.2f}")
            print(f"   Nodo {v}: capacidad={feat_v[1]:.2f}, demanda={feat_v[2]:.2f}, riesgo={feat_v[5]:.2f}")
        
        # Comparar con arista de Kruskal que reemplazó
        if only_kruskal:
            u_k, v_k = list(only_kruskal)[0]
            dist_kruskal = G[u_k][v_k]['weight']
            print(f"   🔄 Reemplazó: ({u_k}, {v_k}): {dist_kruskal:.2f}m")
            print(f"   📊 Sacrificio: +{dist_gnn - dist_kruskal:.2f}m")
    
    print("=" * 80)