# data_loader.py
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Tuple, Dict, List
import config

class GraphDataLoader:
    """
    Cargador de datos para grafos de redes eléctricas.
    Lee archivos CSV/XLSX y convierte a formato PyTorch Geometric.
    """
    
    def __init__(self, file_path: str):
        """
        Inicializa el cargador de datos.
        
        Args:
            file_path: Ruta al archivo CSV o XLSX con los datos del grafo
        """
        self.file_path = Path(file_path)
        self.graph_data = None
        self.nx_graph = None
        
    def load_from_file(self) -> pd.DataFrame:
        """
        Carga los datos desde archivo CSV o XLSX.
        
        Formato esperado del archivo:
        - CSV con aristas: node_from, node_to, weight (distancia en metros)
        - O: node_id, x_coord, y_coord (coordenadas de nodos)
        
        Returns:
            DataFrame con los datos cargados
        """
        if self.file_path.suffix == '.csv':
            df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Formato no soportado: {self.file_path.suffix}")
        
        print(f"✓ Archivo cargado: {self.file_path.name}")
        print(f"  Dimensiones: {df.shape}")
        print(f"  Columnas: {list(df.columns)}")
        
        return df
    
    def build_graph_from_edges(self, df: pd.DataFrame) -> nx.Graph:
        """
        Construye un grafo de NetworkX desde lista de aristas.
        
        Args:
            df: DataFrame con columnas [node_from, node_to, weight]
        
        Returns:
            Grafo de NetworkX
        """
        G = nx.Graph()
        
        # Detectar nombres de columnas (flexibilidad en nombres)
        col_names = df.columns.str.lower()
        
        # Buscar columnas de origen, destino y peso
        from_col = [c for c in df.columns if 'from' in c.lower() or 'origen' in c.lower()][0]
        to_col = [c for c in df.columns if 'to' in c.lower() or 'destino' in c.lower()][0]
        weight_col = [c for c in df.columns if 'weight' in c.lower() or 'distancia' in c.lower() or 'peso' in c.lower()][0]
        
        # Agregar aristas con pesos
        for _, row in df.iterrows():
            G.add_edge(
                int(row[from_col]),
                int(row[to_col]),
                weight=float(row[weight_col])
            )
        
        print(f"✓ Grafo construido:")
        print(f"  Nodos: {G.number_of_nodes()}")
        print(f"  Aristas: {G.number_of_edges()}")
        print(f"  Conexo: {nx.is_connected(G)}")
        
        return G
    
    def build_graph_from_coordinates(self, df_nodes: pd.DataFrame, 
                                     df_edges: pd.DataFrame = None) -> nx.Graph:
        """
        Construye un grafo desde coordenadas de nodos.
        Si no hay aristas definidas, crea grafo completo con distancias euclidianas.
        
        Args:
            df_nodes: DataFrame con [node_id, x_coord, y_coord]
            df_edges: DataFrame opcional con aristas predefinidas
        
        Returns:
            Grafo de NetworkX
        """
        G = nx.Graph()
        
        # Agregar nodos con coordenadas
        for _, row in df_nodes.iterrows():
            node_id = int(row['node_id']) if 'node_id' in row else int(row.iloc[0])
            x = float(row['x_coord']) if 'x_coord' in row else float(row.iloc[1])
            y = float(row['y_coord']) if 'y_coord' in row else float(row.iloc[2])
            G.add_node(node_id, pos=(x, y))
        
        # Si hay aristas predefinidas, usarlas
        if df_edges is not None:
            for _, row in df_edges.iterrows():
                u, v = int(row.iloc[0]), int(row.iloc[1])
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                weight = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
                G.add_edge(u, v, weight=weight)
        else:
            # Crear grafo completo con distancias euclidianas
            nodes = list(G.nodes())
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    pos_u = G.nodes[u]['pos']
                    pos_v = G.nodes[v]['pos']
                    weight = np.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
                    G.add_edge(u, v, weight=weight)
        
        print(f"✓ Grafo desde coordenadas:")
        print(f"  Nodos: {G.number_of_nodes()}")
        print(f"  Aristas: {G.number_of_edges()}")
        
        return G
    
    def to_pyg_data(self, G: nx.Graph) -> Data:
        """
        Convierte grafo de NetworkX a formato PyTorch Geometric.
        
        Args:
            G: Grafo de NetworkX
        
        Returns:
            Objeto Data de PyTorch Geometric
        """
        # Mapear nodos a índices consecutivos
        node_mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
        num_nodes = len(node_mapping)
        
        # Construir edge_index y edge_attr (pesos)
        edge_index = []
        edge_attr = []
        
        for u, v, data in G.edges(data=True):
            idx_u = node_mapping[u]
            idx_v = node_mapping[v]
            weight = data['weight']
            
            # Grafo no dirigido: agregar ambas direcciones
            edge_index.append([idx_u, idx_v])
            edge_index.append([idx_v, idx_u])
            edge_attr.append([weight])
            edge_attr.append([weight])
        
        # Convertir a tensores
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Características de nodos (por ahora, vector de unos)
        x = torch.ones((num_nodes, 1), dtype=torch.float)
        
        # Crear objeto Data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        print(f"✓ Datos convertidos a PyTorch Geometric:")
        print(f"  x: {data.x.shape}")
        print(f"  edge_index: {data.edge_index.shape}")
        print(f"  edge_attr: {data.edge_attr.shape}")
        
        return data
    
    def load_and_process(self) -> Tuple[nx.Graph, Data]:
        """
        Función principal: carga y procesa el grafo completo.
        
        Returns:
            Tupla (grafo NetworkX, datos PyG)
        """
        df = self.load_from_file()
        
        # Detectar tipo de archivo (aristas o coordenadas)
        columns_lower = [c.lower() for c in df.columns]
        
        if any('from' in c or 'to' in c or 'origen' in c or 'destino' in c 
               for c in columns_lower):
            # Archivo de aristas
            G = self.build_graph_from_edges(df)
        else:
            # Archivo de coordenadas
            G = self.build_graph_from_coordinates(df)
        
        self.nx_graph = G
        self.graph_data = self.to_pyg_data(G)
        
        return G, self.graph_data


def load_tibasosa_graph(file_name: str = None) -> Tuple[nx.Graph, Data]:
    """
    Función auxiliar para cargar el grafo de Tibasosa.
    
    Args:
        file_name: Nombre del archivo (opcional, usa config si no se especifica)
    
    Returns:
        Tupla (grafo NetworkX, datos PyG)
    """
    if file_name is None:
        file_name = config.GRAPH_CONFIG['graph_file']
    
    file_path = config.DATA_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {file_path}\n"
            f"Por favor, coloca el archivo en el directorio: {config.DATA_DIR}"
        )
    
    loader = GraphDataLoader(file_path)
    return loader.load_and_process()