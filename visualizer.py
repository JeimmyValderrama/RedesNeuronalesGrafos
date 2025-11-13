"""
visualizer.py
=============
Herramientas de visualización para grafos, MST y análisis de resultados.
Incluye gráficos interactivos y exportación de resultados.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import config


class GraphVisualizer:
    """
    Clase para visualización de grafos y resultados de MST.
    """
    
    def __init__(self, figsize: Tuple = None):
        """
        Inicializa el visualizador.
        
        Args:
            figsize: Tamaño de las figuras (ancho, alto)
        """
        self.figsize = figsize or config.VISUALIZATION_CONFIG['figure_size']
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_graph(self, 
                   G: nx.Graph,
                   pos: Dict = None,
                   title: str = "Grafo",
                   highlight_edges: List = None,
                   save_path: Path = None):
        """
        Visualiza un grafo con opciones de resaltado.
        
        Args:
            G: Grafo de NetworkX
            pos: Diccionario de posiciones de nodos
            title: Título del gráfico
            highlight_edges: Lista de aristas a resaltar
            save_path: Ruta para guardar (opcional)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calcular layout si no se proporciona
        if pos is None:
            if 'pos' in next(iter(G.nodes(data=True)))[1]:
                pos = nx.get_node_attributes(G, 'pos')
            else:
                pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Dibujar todas las aristas en gris claro
        nx.draw_networkx_edges(
            G, pos,
            width=1.0,
            alpha=0.3,
            edge_color='gray',
            ax=ax
        )
        
        # Resaltar aristas específicas (MST)
        if highlight_edges is not None:
            edge_list = [(u, v) for u, v in highlight_edges]
            weights = [G[u][v]['weight'] for u, v in edge_list]
            
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edge_list,
                width=3.0,
                edge_color=weights,
                edge_cmap=plt.cm.Blues,
                edge_vmin=min(weights),
                edge_vmax=max(weights),
                ax=ax
            )
        
        # Dibujar nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_size=config.VISUALIZATION_CONFIG['node_size'],
            node_color=config.VISUALIZATION_CONFIG['node_color'],
            alpha=0.9,
            ax=ax
        )
        
        # Etiquetas de nodos
        nx.draw_networkx_labels(
            G, pos,
            font_size=config.VISUALIZATION_CONFIG['font_size'],
            font_color='white',
            font_weight='bold',
            ax=ax
        )
        
        # Etiquetas de pesos (solo para aristas resaltadas)
        if highlight_edges is not None:
            edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}m" 
                          for u, v in highlight_edges}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8,
                ax=ax
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG['dpi'],
                       bbox_inches='tight')
            print(f"✓ Gráfica guardada en: {save_path}")
        
        plt.show()
    
    def plot_mst_comparison(self,
                           G: nx.Graph,
                           results: Dict,
                           pos: Dict = None,
                           save_path: Path = None):
        """
        Compara visualmente diferentes soluciones de MST.
        
        Args:
            G: Grafo original
            results: Diccionario con resultados de diferentes algoritmos
            pos: Posiciones de nodos
            save_path: Ruta para guardar
        """
        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 6))
        
        if n_algorithms == 1:
            axes = [axes]
        
        # Calcular layout si no se proporciona
        if pos is None:
            if 'pos' in next(iter(G.nodes(data=True)))[1]:
                pos = nx.get_node_attributes(G, 'pos')
            else:
                pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        for ax, (algo_name, result) in zip(axes, results.items()):
            mst = result['mst']
            weight = result['weight']

            # --- Validar y corregir nodos del MST respecto al grafo principal ---
            missing_nodes = [n for n in mst.nodes() if n not in pos]
            if missing_nodes:
                print(f"[⚠️] Corrigiendo nodos faltantes en {algo_name}: {missing_nodes}")
                
                # Si los nodos son índices numéricos y G tiene nombres
                if all(isinstance(n, int) for n in missing_nodes):
                    mapping = {old: list(G.nodes())[old] for old in mst.nodes()}
                    mst = nx.relabel_nodes(mst, mapping)
                else:
                    # Si son nombres, generar layout automático solo para esos nodos
                    extra_pos = nx.spring_layout(mst, seed=42)
                    pos.update(extra_pos)

            
            # Grafo completo en gris
            nx.draw_networkx_edges(
                G, pos,
                width=0.5,
                alpha=0.2,
                edge_color='gray',
                ax=ax
            )
            
            # MST resaltado
            nx.draw_networkx_edges(
                mst, pos,
                width=3.0,
                edge_color='darkblue',
                ax=ax
            )
            
            # Nodos
            nx.draw_networkx_nodes(
                G, pos,
                node_size=400,
                node_color='lightblue',
                edgecolors='darkblue',
                linewidths=2,
                ax=ax
            )
            
            # Etiquetas
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold',
                ax=ax
            )
            
            # Título con información
            title = f"{algo_name.upper()}\n"
            title += f"Peso: {weight:.2f}m"
            if 'optimality_gap' in result:
                title += f"\nGap: {result['optimality_gap']:.2f}%"
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG['dpi'],
                       bbox_inches='tight')
            print(f"✓ Comparación guardada en: {save_path}")
        
        plt.show()
    
    def plot_edge_probabilities(self,
                               G: nx.Graph,
                               edge_probs: Dict[Tuple, float],
                               mst_edges: List[Tuple] = None,
                               pos: Dict = None,
                               save_path: Path = None):
        """
        Visualiza probabilidades predichas por GNN para cada arista.
        
        Args:
            G: Grafo
            edge_probs: Diccionario {(u, v): probabilidad}
            mst_edges: Lista de aristas en MST real (para comparar)
            pos: Posiciones de nodos
            save_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if pos is None:
            if 'pos' in next(iter(G.nodes(data=True)))[1]:
                pos = nx.get_node_attributes(G, 'pos')
            else:
                pos = nx.spring_layout(G, seed=42)
        
        # Preparar colores según probabilidades
        edges = list(G.edges())
        probs = [edge_probs.get((u, v), edge_probs.get((v, u), 0)) 
                for u, v in edges]
        
        # Dibujar aristas coloreadas por probabilidad
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=3.0,
            edge_color=probs,
            edge_cmap=plt.cm.RdYlGn,
            edge_vmin=0,
            edge_vmax=1,
            ax=ax
        )
        
        # Resaltar MST real si se proporciona
        if mst_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=mst_edges,
                width=5.0,
                edge_color='blue',
                style='dashed',
                alpha=0.5,
                ax=ax
            )
        
        # Nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_size=500,
            node_color='lightgray',
            edgecolors='black',
            linewidths=2,
            ax=ax
        )
        
        # Etiquetas
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlGn,
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probabilidad de estar en MST', rotation=270, labelpad=20)
        
        title = "Probabilidades Predichas por GNN"
        if mst_edges:
            title += "\n(Líneas azules punteadas = MST real)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG['dpi'],
                       bbox_inches='tight')
            print(f"✓ Probabilidades guardadas en: {save_path}")
        
        plt.show()
    
    def plot_weight_distribution(self,
                                results: Dict,
                                save_path: Path = None):
        """
        Compara distribución de pesos entre algoritmos.
        
        Args:
            results: Diccionario con resultados
            save_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(results.keys())
        weights = [results[algo]['weight'] for algo in algorithms]
        colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(algorithms)]
        
        bars = ax.bar(algorithms, weights, color=colors, alpha=0.7, edgecolor='black')
        
        # Agregar valores sobre las barras
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight:.2f}m',
                   ha='center', va='bottom', fontweight='bold')
        
        # Línea de referencia (óptimo)
        optimal = min(weights)
        ax.axhline(y=optimal, color='green', linestyle='--', 
                  linewidth=2, label='Óptimo')
        
        ax.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peso Total (metros)', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Peso Total del MST', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION_CONFIG['dpi'])
            print(f"✓ Distribución guardada en: {save_path}")
        
        plt.show()
    
    def export_solution_to_csv(self,
                              mst: nx.Graph,
                              file_path: Path,
                              algorithm_name: str = "MST"):
        """
        Exporta solución MST a CSV.
        
        Args:
            mst: Grafo MST
            file_path: Ruta del archivo CSV
            algorithm_name: Nombre del algoritmo usado
        """
        import pandas as pd
        
        edges = []
        for u, v, data in mst.edges(data=True):
            edges.append({
                'Poste_Origen': u,
                'Poste_Destino': v,
                'Distancia_metros': data['weight'],
                'Algoritmo': algorithm_name
            })
        
        df = pd.DataFrame(edges)
        df = df.sort_values('Distancia_metros')
        
        # Agregar totales
        total_row = {
            'Poste_Origen': 'TOTAL',
            'Poste_Destino': '',
            'Distancia_metros': df['Distancia_metros'].sum(),
            'Algoritmo': algorithm_name
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        df.to_csv(file_path, index=False)
        print(f"✓ Solución exportada a: {file_path}")
        
        return df


# ============================================================================
# EJEMPLO DE USO
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VISUALIZADOR DE GRAFOS MST")
    print("=" * 70)
    
    # Crear grafo de ejemplo
    G = nx.Graph()
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2),
        (2, 3, 4), (3, 4, 2), (4, 5, 6), (2, 5, 5)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    # Calcular MST
    mst = nx.minimum_spanning_tree(G, weight='weight')
    
    # Visualizar
    visualizer = GraphVisualizer()
    
    print("\n1. Visualizando grafo original...")
    visualizer.plot_graph(
        G,
        title="Grafo Original - Red Eléctrica",
        save_path=config.PLOTS_DIR / 'grafo_original.png'
    )
    
    print("\n2. Visualizando MST...")
    visualizer.plot_graph(
        G,
        title="Minimum Spanning Tree",
        highlight_edges=list(mst.edges()),
        save_path=config.PLOTS_DIR / 'mst_solution.png'
    )
    
    print("\n3. Exportando solución a CSV...")
    visualizer.export_solution_to_csv(
        mst,
        config.RESULTS_DIR / 'solucion_mst.csv',
        algorithm_name='Kruskal'
    )
    
    print("\n✓ Visualización completada!")