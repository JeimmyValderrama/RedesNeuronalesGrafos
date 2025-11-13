"""
tibasosa_coordinates_loader.py - VERSIÓN CORREGIDA
====================================================
Corrige indexación de nodos y usa codificador automático de características.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Tuple, Dict
from geopy.distance import geodesic
import config
from feature_encoder import AutoFeatureEncoder


class TibasosaCoordinatesLoader:
    """
    Cargador especializado para coordenadas geográficas de Tibasosa.
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None
        self.G = None
        self.data = None
        self.node_info = {}
        self.feature_encoder = AutoFeatureEncoder()
    
    def load_coordinates(self):
        """Carga el archivo CSV de coordenadas geográficas."""
        print("=" * 70)
        print("CARGANDO COORDENADAS GEOGRÁFICAS DE TIBASOSA")
        print("=" * 70 + "\n")

        try:
            df = pd.read_csv(self.file_path, encoding='latin1', sep=';', skip_blank_lines=True)
            print(f"✓ Archivo cargado: {self.file_path}")
        except Exception as e:
            print(f"❌ Error al leer el archivo: {e}")
            return None

        print(f"  Puntos encontrados: {len(df)}")
        print(f"  Columnas: {list(df.columns)}")

        # Normalizar nombres de columnas
        df.columns = df.columns.str.strip().str.lower()
        
        # Detectar columnas de coordenadas (flexibilidad en nombres)
        lat_col = None
        lon_col = None
        
        for col in df.columns:
            if 'lat' in col:
                lat_col = col
            elif 'lon' in col:
                lon_col = col
        
        if lat_col is None or lon_col is None:
            raise ValueError("No se encontraron columnas de latitud/longitud")
        
        # Renombrar para consistencia
        df = df.rename(columns={lat_col: 'latitud', lon_col: 'longitud'})

        # Convertir coordenadas a numérico
        df['latitud'] = df['latitud'].astype(str).str.replace(',', '.', regex=False)
        df['longitud'] = df['longitud'].astype(str).str.replace(',', '.', regex=False)
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')

        # Eliminar filas inválidas
        df = df.dropna(subset=['latitud', 'longitud'])
        
        # Detectar columna de nombre (flexible)
        name_col = None
        for col in df.columns:
            if 'nombre' in col or 'name' in col:
                name_col = col
                break
        
        if name_col and name_col != 'nombre':
            df = df.rename(columns={name_col: 'nombre'})

        print("\nRango de coordenadas:")
        print(f"  Latitud:  [{df['latitud'].min():.6f}, {df['latitud'].max():.6f}]")
        print(f"  Longitud: [{df['longitud'].min():.6f}, {df['longitud'].max():.6f}]\n")
        
        self.df = df
        return df
    
    def calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """Calcula distancia geodésica en metros."""
        return geodesic(coord1, coord2).meters
    
    def build_complete_graph(self, max_distance: float = None) -> nx.Graph:
        """
        Construye grafo completo con indexación desde 0 (CORREGIDO).
        """
        G = nx.Graph()
        
        print("\n" + "=" * 70)
        print("CONSTRUYENDO GRAFO DE RED ELÉCTRICA")
        print("=" * 70)
        
        # 🔴 CAMBIO CRÍTICO: Nodos empiezan desde 0
        for idx, row in self.df.iterrows():
            node_id = idx  # Índice directo desde 0
            
            node_attrs = {
                'pos': (row['latitud'], row['longitud']),
                'name': row.get('nombre', f"Punto_{node_id}"),
                'lat': row['latitud'],
                'lon': row['longitud']
            }
            
            # Agregar todas las columnas adicionales automáticamente
            for col in self.df.columns:
                if col not in ['latitud', 'longitud', 'nombre', 'pos']:
                    node_attrs[col] = row[col]
            
            G.add_node(node_id, **node_attrs)
            self.node_info[node_id] = node_attrs
        
        # Calcular distancias
        nodes = list(G.nodes())
        total_edges = 0
        edges_added = 0
        
        print(f"\nCalculando distancias entre {len(nodes)} puntos...")
        
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                total_edges += 1
                
                coord_u = G.nodes[u]['pos']
                coord_v = G.nodes[v]['pos']
                distance = self.calculate_distance(coord_u, coord_v)
                
                if max_distance is None or distance <= max_distance:
                    G.add_edge(u, v, weight=distance)
                    edges_added += 1
        
        print(f"\n✓ Grafo construido exitosamente:")
        print(f"  • Nodos (índices 0-{len(nodes)-1}): {G.number_of_nodes()}")
        print(f"  • Aristas posibles: {total_edges}")
        print(f"  • Aristas incluidas: {edges_added}")
        print(f"  • Grafo conexo: {nx.is_connected(G)}")
        
        if not nx.is_connected(G):
            print("\n⚠ ADVERTENCIA: El grafo NO es conexo!")
        
        if edges_added > 0:
            weights = [d['weight'] for _, _, d in G.edges(data=True)]
            print(f"\nEstadísticas de distancias:")
            print(f"  • Mínima: {min(weights):.2f} metros")
            print(f"  • Máxima: {max(weights):.2f} metros")
            print(f"  • Promedio: {np.mean(weights):.2f} metros")
            print(f"  • Mediana: {np.median(weights):.2f} metros")
        
        self.G = G
        return G
    
    def to_pyg_data(self) -> Data:
        """
        Convierte grafo a formato PyTorch Geometric con características automáticas.
        """
        if self.G is None:
            raise ValueError("Primero construye el grafo con build_complete_graph()")
        
        # 🔴 CAMBIO: Mapeo directo sin transformación
        node_mapping = {node: node for node in sorted(self.G.nodes())}
        num_nodes = len(node_mapping)
        
        # Construir edge_index y edge_attr
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.G.edges(data=True):
            weight = data['weight']
            
            edge_index.append([u, v])
            edge_index.append([v, u])
            edge_attr.append([weight])
            edge_attr.append([weight])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 🔥 CARACTERÍSTICAS AUTOMÁTICAS
        print("\n" + "=" * 70)
        print("CODIFICANDO CARACTERÍSTICAS DE NODOS")
        print("=" * 70)
        
        # Entrenar codificador con el DataFrame
        node_features = self.feature_encoder.fit_transform(self.df)
        
        # Agregar grado normalizado como primera característica
        degrees = dict(self.G.degree())
        degree_features = np.array([[degrees[i] / num_nodes] for i in range(num_nodes)])
        node_features = np.hstack([degree_features, node_features])
        
        # Agregar nombre de característica de grado
        self.feature_encoder.feature_names.insert(0, 'degree_norm')
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        self.data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        print(f"\n✓ Datos convertidos a PyTorch Geometric:")
        print(f"  • x (características): {self.data.x.shape}")
        print(f"  • Características por nodo: {x.shape[1]}")
        print(f"  • edge_index: {self.data.edge_index.shape}")
        print(f"  • edge_attr: {self.data.edge_attr.shape}")
        
        return self.data
    
    def export_graph_to_csv(self, output_path: str):
        """Exporta el grafo a CSV."""
        if self.G is None:
            raise ValueError("Primero construye el grafo")
        
        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({
                'node_from': u,
                'node_to': v,
                'weight': data['weight'],
                'nombre_from': self.G.nodes[u]['name'],
                'nombre_to': self.G.nodes[v]['name']
            })
        
        df_edges = pd.DataFrame(edges)
        df_edges.to_csv(output_path, index=False)
        print(f"\n✓ Grafo exportado a: {output_path}")
    
    def visualize_on_map(self, save_path: str = None):
        """Crea visualización geográfica."""
        if self.G is None:
            raise ValueError("Primero construye el grafo")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = {node: (data['lon'], data['lat']) 
               for node, data in self.G.nodes(data=True)}
        
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, width=1, ax=ax)
        nx.draw_networkx_nodes(self.G, pos, node_size=300, 
                              node_color='red', alpha=0.7, ax=ax)
        
        labels = {node: data['name'].split()[0] 
                 for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8, ax=ax)
        
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.set_title('Red Eléctrica Tibasosa - Ubicación Geográfica', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Mapa guardado en: {save_path}")
        
        plt.show()


def load_tibasosa_coordinates(file_name: str = 'tibasosa_coordenadas.csv',
                              max_distance: float = None) -> Tuple[nx.Graph, Data]:
    """
    Función simplificada para cargar coordenadas de Tibasosa.
    """
    file_path = config.DATA_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {file_path}\n"
            f"Coloca tu archivo de coordenadas en: {config.DATA_DIR}"
        )
    
    loader = TibasosaCoordinatesLoader(file_path)
    loader.load_coordinates()
    G = loader.build_complete_graph(max_distance=max_distance)
    data = loader.to_pyg_data()
    
    return G, data