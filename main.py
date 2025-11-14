"""
main.py - VERSIÓN CORREGIDA
============================
Coordina entrenamiento con características dinámicas.
"""

import argparse
import torch
import networkx as nx
from pathlib import Path
import gc 

import config
# Limitar hilos de PyTorch (evita uso excesivo de memoria)
torch.set_num_threads(1)

# Forzar recolección de basura antes de entrenar / evaluar
gc.collect()
from data_loader import load_tibasosa_graph
from tibasosa_coordinates_loader import load_tibasosa_coordinates
from dataset_generator import (
    generate_synthetic_graphs,
    split_dataset,
    save_dataset,
    load_dataset
)
from model_gat import EdgeGAT, model_summary
from trainer import MST_Trainer
from mst_algorithms import (
    compare_mst_algorithms,
    print_comparison_results,
    extract_mst_solution,
    validate_mst,
    analyze_decision_differences
)
from visualizer import GraphVisualizer


def generate_and_save_dataset(analyzer=None):
    """
    Genera dataset sintético adaptado al dataset objetivo.
    
    Args:
        analyzer: DatasetAnalyzer con esquema del dataset objetivo (None = modo básico)
    """
    print("\n" + "=" * 70)
    print("FASE 1: GENERACIÓN DE DATASET SINTÉTICO")
    print("=" * 70)
    
    num_graphs = config.DATASET_CONFIG['num_graphs']
    seed = config.DATASET_CONFIG['seed']
    
    if analyzer is not None:
        print(f"\n Generando {num_graphs} grafos ADAPTADOS al dataset objetivo...")
        print(f"   Características: {analyzer.get_num_features()}")
    else:
        print(f"\n Generando {num_graphs} grafos en MODO BÁSICO...")
        print(f"   (sin analizador - solo grado normalizado)")
    
    data_list = generate_synthetic_graphs(
        num_graphs, 
        analyzer=analyzer,
        seed=seed
    )
    
    print("\nDividiendo dataset...")
    train_data, val_data, test_data = split_dataset(data_list, seed=seed)
    
    print("\nGuardando datasets...")
    save_dataset(train_data, 'train_dataset.pkl')
    save_dataset(val_data, 'val_dataset.pkl')
    save_dataset(test_data, 'test_dataset.pkl')
    
    print("\n✓ Dataset generado y guardado exitosamente!")
    return train_data, val_data, test_data


def train_model(train_data, val_data):
    """Entrena el modelo GAT."""
    print("\n" + "=" * 70)
    print("FASE 2: ENTRENAMIENTO DEL MODELO GAT")
    print("=" * 70)
    
    print("\nCreando modelo GAT...")
    model = EdgeGAT()
    
    # INICIALIZAR MODELO CON PRIMER BATCH
    print("Inicializando modelo con datos de muestra...")
    sample_data = train_data[0].to(config.DEVICE)
    _ = model(sample_data)
    
    model_summary(model)
    
    print("\nInicializando entrenador...")
    trainer = MST_Trainer(model, train_data, val_data)
    
    print("\nComenzando entrenamiento...")
    history = trainer.train(num_epochs=config.TRAINING_CONFIG['num_epochs'])
    
    print("\nGenerando gráficas de entrenamiento...")
    plot_path = config.PLOTS_DIR / 'training_history.png'
    trainer.plot_history(save_path=plot_path)
    
    print("\n✓ Entrenamiento completado!")
    return model, trainer


def evaluate_model(model, test_data):
    """Evalúa el modelo."""
    print("\n" + "=" * 70)
    print("FASE 3: EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    from torch_geometric.loader import DataLoader
    
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    model.eval()
    model = model.to(config.DEVICE)
    
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    print("\nEvaluando en conjunto de prueba...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(config.DEVICE)
            logits = model(batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            total_correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    import numpy as np
    labels = np.array(all_labels)
    preds = np.array(all_preds)
    
    accuracy = total_correct / total_samples
    
    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print("\nRESULTADOS EN TEST SET:")
    print(f"  • Accuracy: {accuracy:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall: {recall:.4f}")
    print(f"  • F1-Score: {f1:.4f}")
    
    print("\n✓ Evaluación completada!")


def solve_tibasosa_problem(model=None):
    """Resuelve el problema de Tibasosa."""
    print("\n" + "=" * 70)
    print("FASE 4: SOLUCIÓN DEL PROBLEMA DE TIBASOSA")
    print("=" * 70)
    
    print("\nCargando grafo de Tibasosa con coordenadas reales...")
    try:
        G, data = load_tibasosa_coordinates('tibasosa_coordenadas.csv')
    except FileNotFoundError as e:
        print(f"\n⚠ ERROR: {e}")
        return
    
    # Comparar algoritmos
    print("\nComparando algoritmos MST...")
    results = compare_mst_algorithms(G, model, data)
    print_comparison_results(results)
    
    # ANÁLISIS DE DECISIONES DIFERENTES
    if 'gnn_smart' in results:
        analyze_decision_differences(G, results, data)
    
    # Validar soluciones
    print("\nValidando soluciones...")
    for algo_name, result in results.items():
        print(f"\n{algo_name.upper()}:")
        validation = validate_mst(result['mst'], G)
        for check, passed in validation.items():
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {check}")
    
    # Visualizar
    print("\nGenerando visualizaciones...")
    visualizer = GraphVisualizer()
    
    visualizer.plot_graph(
        G,
        title="Red Eléctrica Tibasosa - Grafo Completo",
        save_path=config.PLOTS_DIR / 'tibasosa_grafo_completo.png'
    )
    
    visualizer.plot_mst_comparison(
        G,
        results,
        save_path=config.PLOTS_DIR / 'tibasosa_mst_comparison.png'
    )
    
    visualizer.plot_weight_distribution(
        results,
        save_path=config.PLOTS_DIR / 'tibasosa_weight_comparison.png'
    )
    
    # Exportar soluciones
    print("\nExportando soluciones a CSV...")
    for algo_name, result in results.items():
        file_name = f'solucion_tibasosa_{algo_name}.csv'
        visualizer.export_solution_to_csv(
            result['mst'],
            config.RESULTS_DIR / file_name,
            algorithm_name=algo_name
        )
    
    # MOSTRAR SOLUCIÓN ÓPTIMA (KRUSKAL)
    print("\n" + "=" * 70)
    print("SOLUCIÓN KRUSTAL")
    print("=" * 70)
    solution_kruskal = extract_mst_solution(results['kruskal']['mst'])
    print("\nConexiones de cable necesarias:")
    for i, (u, v, w) in enumerate(solution_kruskal, 1):
        print(f"  {i}. Poste {u} → Poste {v}: {w:.2f} metros")
    
    total_weight_kruskal = results['kruskal']['weight']
    print(f"\n{'='*70}")
    print(f"METRAJE TOTAL DE CABLE (KRUSKAL): {total_weight_kruskal:.2f} METROS")
    print(f"{'='*70}")
    
    # MOSTRAR SOLUCIÓN GNN SI DISPONIBLE
    if 'gnn_smart' in results:
        print("\n" + "=" * 70)
        print("SOLUCIÓN GNN-GUIDED")
        print("=" * 70)
        solution_gnn = extract_mst_solution(results['gnn_smart']['mst'])
        print("\nConexiones de cable necesarias (GNN):")
        for i, (u, v, w) in enumerate(solution_gnn, 1):
            print(f"  {i}. Poste {u} → Poste {v}: {w:.2f} metros")
        
        gnn_weight = results['gnn_smart']['weight']
        gap = results['gnn_smart']['optimality_gap']
        
        print(f"\n{'='*70}")
        print(f"METRAJE TOTAL DE CABLE (GNN): {gnn_weight:.2f} METROS")
        print(f"{'='*70}")
        
        print(f"\n🤖 ANÁLISIS COMPARATIVO:")
        print(f"   Kruskal (óptimo):   {total_weight_kruskal:.2f} metros")
        print(f"   GNN (inteligente):  {gnn_weight:.2f} metros")
        print(f"   Diferencia:         {gnn_weight - total_weight_kruskal:+.2f} metros ({gap:+.2f}%)")
        
        if abs(gap) < 0.01:
            print(f"\n   ¡GNN ENCONTRÓ LA SOLUCIÓN ÓPTIMA!")
        elif gap < 5:
            print(f"\n   GNN logró solución excelente (< 5% gap)")
            print(f"    Ventajas: prioriza nodos críticos, minimiza riesgo, optimiza demanda")
    else:
        print("\n No hay solución GNN disponible (modelo no cargado)")
    
    print("\n✓ Problema resuelto exitosamente!")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Optimización de Red Eléctrica usando GNN'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'solve', 'full'],
        default='full',
        help='Modo de ejecución (default: full)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Saltar entrenamiento y cargar modelo existente'
    )
    parser.add_argument(
        '--regenerate-dataset',
        action='store_true',
        help='Forzar regeneración del dataset'
    )
    parser.add_argument(
        '--dataset-file',
        type=str,
        default='tibasosa_coordenadas.csv',
        help='Archivo CSV del dataset objetivo'
    )
    
    args = parser.parse_args()
    
    # PASO 1: ANALIZAR DATASET OBJETIVO
    from dataset_analyzer import analyze_target_dataset
    
    analyzer = None
    num_features = 1  # Default
    
    try:
        print(f"\n{'='*70}")
        print("ANALIZANDO DATASET OBJETIVO")
        print(f"{'='*70}")
        
        analyzer = analyze_target_dataset(args.dataset_file)
        num_features = analyzer.get_num_features()
        
        print(f"\n DATASET ANALIZADO EXITOSAMENTE")
        print(f"   • Características totales: {num_features}")
        print(f"   • El dataset sintético se adaptará a estas características")
        
    except Exception as e:
        print(f"\n No se pudo analizar dataset objetivo: {e}")
        print(f"   • Usando características por defecto")
        analyzer = None
    
    #  PASO 2: CARGAR DATOS REALES
    _, data_tibasosa = load_tibasosa_coordinates(args.dataset_file)
    actual_features = data_tibasosa.x.shape[1]
    
    model = None
    
    if args.mode in ['train', 'full']:
        train_path = config.DATA_DIR / 'train_dataset.pkl'
        val_path = config.DATA_DIR / 'val_dataset.pkl'
        
        # Verificar si necesita regenerar
        need_regenerate = args.regenerate_dataset
        
        if train_path.exists() and val_path.exists() and not need_regenerate:
            print("\n✓ Datasets encontrados, verificando compatibilidad...")
            train_data = load_dataset('train_dataset.pkl')
            val_data = load_dataset('val_dataset.pkl')
            
            # Verificar dimensiones
            if train_data[0].x.shape[1] != actual_features:
                print(f"\n Dimensiones incompatibles:")
                print(f"   Dataset sintético: {train_data[0].x.shape[1]} features")
                print(f"   Dataset objetivo: {actual_features} features")
                print(f"   Regenerando...")
                need_regenerate = True
            else:
                print(f" Dimensiones compatibles ({actual_features} features)")
        else:
            need_regenerate = True
        
        if need_regenerate:
            print("\n Generando dataset sintético adaptado...")
            train_data, val_data, test_data = generate_and_save_dataset(analyzer=analyzer)
        
        if not args.skip_training:
            model, trainer = train_model(train_data, val_data)
        else:
            print("\n Saltar entrenamiento - Cargando modelo existente...")
            model = EdgeGAT()
            model_path = config.MODELS_DIR / 'best_model.pt'
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Modelo cargado desde: {model_path}")
            else:
                print(f" No se encontró modelo guardado")
                model = None
    
    if args.mode in ['evaluate', 'full']:
        test_path = config.DATA_DIR / 'test_dataset.pkl'
        if test_path.exists():
            test_data = load_dataset('test_dataset.pkl')
            if model is None:
                model = EdgeGAT()
                model_path = config.MODELS_DIR / 'best_model.pt'
                checkpoint = torch.load(model_path, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
            evaluate_model(model, test_data)
        else:
            print("\n Test dataset no encontrado, saltando evaluación...")
    
    if args.mode in ['solve', 'full']:
        if model is None:
            print("\n Intentando cargar modelo para resolver problema...")
            model = EdgeGAT()
            model_path = config.MODELS_DIR / 'best_model.pt'
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Modelo cargado desde: {model_path}")
        
        solve_tibasosa_problem(model)
    
    print("\n" + "=" * 70)
    print("¡EJECUCIÓN COMPLETADA!")
    print("=" * 70)
    print("\nArchivos generados:")
    print(f"  • Modelos: {config.MODELS_DIR}")
    print(f"  • Gráficas: {config.PLOTS_DIR}")
    print(f"  • Resultados: {config.RESULTS_DIR}")
    print("\n✓ Proyecto finalizado exitosamente!")


if __name__ == "__main__":
    main()