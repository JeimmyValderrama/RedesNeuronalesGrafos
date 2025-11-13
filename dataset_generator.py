"""
dataset_generator.py - VERSIÓN UNIFICADA INTELIGENTE
=====================================================
Se adapta automáticamente a CUALQUIER dataset y genera características correlacionadas.
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import pickle
from tqdm import tqdm
import config
from dataset_analyzer import DatasetAnalyzer


def generate_smart_features(num_nodes: int, 
                            degrees: Dict[int, int],
                            G: nx.Graph,
                            analyzer: DatasetAnalyzer) -> np.ndarray:
    """
    Genera características con CORRELACIONES REALISTAS basadas en topología.
    
    Correlaciones implementadas:
    - Nodos con alto grado → Mayor capacidad y demanda
    - Nodos centrales (alta betweenness) → Mayor criticidad
    - Alta demanda → Mayor costo de mantenimiento
    - Nodos periféricos → Mayor factor de riesgo
    """
    if not analyzer or not analyzer.analyzed:
        # Fallback: solo grado normalizado
        return np.array([[degrees.get(i, 0) / num_nodes] for i in range(num_nodes)])
    
    features_list = []
    
    # 1. Grado normalizado (siempre primero)
    max_degree = max(degrees.values()) if degrees else 1
    degree_features = np.array([[degrees.get(i, 0) / max_degree] for i in range(num_nodes)])
    features_list.append(degree_features)
    
    # Calcular métricas de centralidad para correlaciones
    try:
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        clustering = nx.clustering(G)
    except:
        betweenness = {i: 0.5 for i in range(num_nodes)}
        closeness = {i: 0.5 for i in range(num_nodes)}
        clustering = {i: 0.5 for i in range(num_nodes)}
    
    # Normalizar métricas
    max_between = max(betweenness.values()) if betweenness else 1
    max_close = max(closeness.values()) if closeness else 1
    
    betweenness_norm = {k: v/max_between for k, v in betweenness.items()}
    closeness_norm = {k: v/max_close for k, v in closeness.items()}
    
    # 2. CARACTERÍSTICAS NUMÉRICAS CON CORRELACIONES
    numeric_values = {}
    
    for col in analyzer.schema['numeric_features']:
        info = analyzer.schema['feature_ranges'][col]
        mean = info['mean']
        std = info['std']
        min_val = info['min']
        max_val = info['max']
        
        values = np.zeros(num_nodes)
        
        for i in range(num_nodes):
            degree_norm = degrees.get(i, 0) / max_degree
            between_norm = betweenness_norm.get(i, 0.5)
            close_norm = closeness_norm.get(i, 0.5)
            
            # Índice de centralidad combinado
            centrality = (between_norm * 0.4 + close_norm * 0.4 + degree_norm * 0.2)
            
            # LÓGICA ESPECÍFICA POR TIPO
            if 'capacidad' in col.lower():
                factor = 0.5 + centrality * 0.5
                base = min_val + (max_val - min_val) * factor
                noise = np.random.normal(0, std * 0.2)
                values[i] = np.clip(base + noise, min_val, max_val)
            
            elif 'demanda' in col.lower():
                if 'capacidad' in numeric_values:
                    cap_factor = numeric_values['capacidad'][i] / max_val
                    factor = cap_factor * 0.6 + centrality * 0.4
                else:
                    factor = centrality
                base = min_val + (max_val - min_val) * factor
                noise = np.random.normal(0, std * 0.25)
                values[i] = np.clip(base + noise, min_val, max_val)
            
            elif 'costo' in col.lower():
                if 'capacidad' in numeric_values or 'demanda' in numeric_values:
                    cap_val = numeric_values.get('capacidad', numeric_values.get('demanda', mean))
                    cap_norm = (cap_val[i] - min_val) / (max_val - min_val + 1e-8)
                    factor = 0.3 + cap_norm * 0.7
                else:
                    factor = 0.5 + centrality * 0.5
                base = min_val + (max_val - min_val) * factor
                noise = np.random.normal(0, std * 0.15)
                values[i] = np.clip(base + noise, min_val, max_val)
            
            elif 'tiempo' in col.lower():
                factor = 1.0 - degree_norm * 0.4
                base = min_val + (max_val - min_val) * factor
                noise = np.random.normal(0, std * 0.3)
                values[i] = np.clip(base + noise, min_val, max_val)
            
            elif 'riesgo' in col.lower():
                factor = 1.0 - centrality
                base = min_val + (max_val - min_val) * factor
                noise = np.random.normal(0, std * 0.2)
                values[i] = np.clip(base + noise, min_val, max_val)
            
            else:
                base = np.random.normal(mean, std)
                values[i] = np.clip(base, min_val, max_val)
        
        col_key = col.split('_')[0]
        numeric_values[col_key] = values.copy()
        
        # Normalizar a [0, 1]
        if max_val > min_val:
            values_norm = (values - min_val) / (max_val - min_val)
        else:
            values_norm = np.ones(num_nodes) * 0.5
        
        features_list.append(values_norm.reshape(-1, 1))
    
    # 3. CARACTERÍSTICAS CATEGÓRICAS CON CORRELACIONES
    for col in analyzer.schema['categorical_features']:
        dist = analyzer.schema['categorical_distributions'][col]
        categories = dist['categories']
        probabilities = np.array(dist['probabilities'])
        num_categories = len(categories)
        
        one_hot = np.zeros((num_nodes, num_categories))
        
        for i in range(num_nodes):
            degree_norm = degrees.get(i, 0) / max_degree
            between_norm = betweenness_norm.get(i, 0.5)
            centrality = (between_norm * 0.6 + degree_norm * 0.4)
            
            adjusted_probs = probabilities.copy()
            
            # AJUSTAR PROBABILIDADES SEGÚN CARACTERÍSTICAS
            if 'tipo' in col.lower():
                if centrality > 0.6:
                    for idx, cat in enumerate(categories):
                        if cat in ['comercial', 'publico']:
                            adjusted_probs[idx] *= 1.5
                        elif cat == 'residencial':
                            adjusted_probs[idx] *= 0.7
                elif centrality < 0.3:
                    for idx, cat in enumerate(categories):
                        if cat == 'residencial':
                            adjusted_probs[idx] *= 1.4
            
            elif 'prioridad' in col.lower():
                if centrality > 0.5:
                    for idx, cat in enumerate(categories):
                        if cat == 'alta':
                            adjusted_probs[idx] *= 1.6
                        elif cat == 'media':
                            adjusted_probs[idx] *= 0.8
            
            elif 'criticidad' in col.lower():
                if between_norm > 0.7:
                    for idx, cat in enumerate(categories):
                        if cat == 'critica':
                            adjusted_probs[idx] *= 2.0
                        elif cat == 'alta':
                            adjusted_probs[idx] *= 1.3
                        elif cat == 'baja':
                            adjusted_probs[idx] *= 0.3
                elif between_norm < 0.3:
                    for idx, cat in enumerate(categories):
                        if cat in ['baja', 'media']:
                            adjusted_probs[idx] *= 1.5
                        elif cat == 'critica':
                            adjusted_probs[idx] *= 0.4
            
            elif 'zona' in col.lower():
                if centrality > 0.5:
                    for idx, cat in enumerate(categories):
                        if cat in ['comercial', 'urbana']:
                            adjusted_probs[idx] *= 1.5
            
            elif 'accesibilidad' in col.lower():
                if degree_norm > 0.6:
                    for idx, cat in enumerate(categories):
                        if cat == 'facil':
                            adjusted_probs[idx] *= 1.6
                        elif cat == 'dificil':
                            adjusted_probs[idx] *= 0.5
                elif degree_norm < 0.3:
                    for idx, cat in enumerate(categories):
                        if cat == 'dificil':
                            adjusted_probs[idx] *= 1.5
            
            elif 'disponibilidad' in col.lower():
                if degree_norm > 0.5:
                    for idx, cat in enumerate(categories):
                        if cat == 'alta':
                            adjusted_probs[idx] *= 1.4
            
            # Normalizar
            adjusted_probs /= (adjusted_probs.sum() + 1e-8)
            
            chosen_idx = np.random.choice(len(categories), p=adjusted_probs)
            one_hot[i, chosen_idx] = 1.0
        
        features_list.append(one_hot)
    
    features = np.hstack(features_list)
    return features


def generate_single_graph(num_nodes: int, num_edges: int, seed: int = None) -> nx.Graph:
    """Genera grafo sintético con modelo Watts-Strogatz."""
    if seed is not None:
        np.random.seed(seed)
    
    k = max(4, min(num_nodes - 1, int(2 * num_edges / num_nodes)))
    k = k if k % 2 == 0 else k - 1
    
    G = nx.connected_watts_strogatz_graph(n=num_nodes, k=k, p=0.3, seed=seed)
    
    current_edges = G.number_of_edges()
    
    if current_edges < num_edges:
        nodes = list(G.nodes())
        attempts = 0
        while G.number_of_edges() < num_edges and attempts < 1000:
            u, v = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
            attempts += 1
    
    elif current_edges > num_edges:
        edges = list(G.edges())
        np.random.shuffle(edges)
        for u, v in edges:
            if G.number_of_edges() <= num_edges:
                break
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v)
    
    # Asignar pesos con variabilidad realista
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    for u, v in G.edges():
        base_weight = np.random.uniform(150, 1000)
        
        # Ajuste por centralidad
        degree_u_norm = degrees[u] / max_degree
        degree_v_norm = degrees[v] / max_degree
        avg_degree_norm = (degree_u_norm + degree_v_norm) / 2
        degree_factor = 1.0 - avg_degree_norm * 0.3
        
        weight = base_weight * degree_factor
        G[u][v]['weight'] = weight
    
    return G


def compute_mst(G: nx.Graph) -> nx.Graph:
    """Calcula MST con Kruskal."""
    return nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')


def graph_to_pyg_data(G: nx.Graph, mst: nx.Graph, 
                      analyzer: DatasetAnalyzer = None) -> Data:
    """
    Convierte grafo a PyTorch Geometric con características INTELIGENTES.
    """
    node_mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
    num_nodes = len(node_mapping)
    
    # Construir edge_index, edge_attr y labels
    edge_index = []
    edge_attr = []
    edge_labels = []
    
    mst_edges = set()
    for u, v in mst.edges():
        mst_edges.add((min(u, v), max(u, v)))
    
    for u, v, data in G.edges(data=True):
        idx_u = node_mapping[u]
        idx_v = node_mapping[v]
        weight = data['weight']
        
        is_in_mst = (min(u, v), max(u, v)) in mst_edges
        
        edge_index.append([idx_u, idx_v])
        edge_index.append([idx_v, idx_u])
        edge_attr.append([weight])
        edge_attr.append([weight])
        edge_labels.append(1.0 if is_in_mst else 0.0)
        edge_labels.append(1.0 if is_in_mst else 0.0)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(edge_labels, dtype=torch.float)
    
    # 🔥 CARACTERÍSTICAS INTELIGENTES CON CORRELACIONES
    if analyzer is not None and analyzer.analyzed:
        degrees = dict(G.degree())
        node_features = generate_smart_features(num_nodes, degrees, G, analyzer)
    else:
        # Fallback: características básicas
        node_features = []
        degrees = dict(G.degree())
        
        for node in sorted(G.nodes()):
            degree_norm = degrees[node] / num_nodes
            node_features.append([degree_norm])
        
        node_features = np.array(node_features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    mst_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
    total_weight = sum(data['weight'] for _, _, data in G.edges(data=True))
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes,
        mst_weight=mst_weight,
        total_weight=total_weight
    )


def generate_synthetic_graphs(num_graphs: int, 
                              min_nodes: int = None,
                              max_nodes: int = None,
                              min_edges: int = None,
                              max_edges: int = None,
                              analyzer: DatasetAnalyzer = None,
                              seed: int = None) -> List[Data]:
    """
    Genera dataset sintético 100% ADAPTATIVO con correlaciones realistas.
    """
    min_nodes = min_nodes or config.DATASET_CONFIG['min_nodes']
    max_nodes = max_nodes or config.DATASET_CONFIG['max_nodes']
    min_edges = min_edges or config.DATASET_CONFIG['min_edges']
    max_edges = max_edges or config.DATASET_CONFIG['max_edges']
    
    if seed is not None:
        np.random.seed(seed)
    
    data_list = []
    
    print(f"\nGenerando grafos...")
    
    for i in tqdm(range(num_graphs), desc="Progreso"):
        n = np.random.randint(min_nodes, max_nodes + 1)
        e = np.random.randint(min_edges, max_edges + 1)
        e = max(e, n)
        
        G = generate_single_graph(n, e, seed=seed + i if seed else None)
        mst = compute_mst(G)
        
        data = graph_to_pyg_data(G, mst, analyzer=analyzer)
        data_list.append(data)
    
    print(f"\n✓ {len(data_list)} grafos generados exitosamente!")
    
    if analyzer:
        print(f"✓ Características inteligentes alineadas con dataset objetivo")
        print(f"✓ Dimensión: {data_list[0].x.shape}")
    
    return data_list


def split_dataset(data_list: List[Data], 
                  train_ratio: float = None,
                  val_ratio: float = None,
                  test_ratio: float = None,
                  seed: int = None) -> Tuple[List[Data], List[Data], List[Data]]:
    """Divide dataset."""
    train_ratio = train_ratio or config.DATASET_CONFIG['train_split']
    val_ratio = val_ratio or config.DATASET_CONFIG['val_split']
    test_ratio = test_ratio or config.DATASET_CONFIG['test_split']
    
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.random.permutation(len(data_list))
    n = len(data_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = [data_list[i] for i in indices[:train_end]]
    val_data = [data_list[i] for i in indices[train_end:val_end]]
    test_data = [data_list[i] for i in indices[val_end:]]
    
    print(f"\n✓ Dataset dividido:")
    print(f"  Entrenamiento: {len(train_data)} grafos")
    print(f"  Validación: {len(val_data)} grafos")
    print(f"  Prueba: {len(test_data)} grafos")
    
    return train_data, val_data, test_data


def save_dataset(data_list: List[Data], file_name: str):
    """Guarda dataset."""
    file_path = config.DATA_DIR / file_name
    with open(file_path, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"✓ Dataset guardado en: {file_path}")


def load_dataset(file_name: str) -> List[Data]:
    """Carga dataset."""
    file_path = config.DATA_DIR / file_name
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"✓ Dataset cargado desde: {file_path}")
    return data_list