"""
regenerate_fixed.py - REGENERA TODO DESDE CERO CORRECTAMENTE
============================================================
"""

import torch
from pathlib import Path
import shutil
import os

# Limpiar archivos viejos
def clean_old_files():
    print("Limpiando archivos anteriores...")
    
    # Archivos individuales a eliminar
    files_to_remove = [
        'dataset_schema.json',
        'test_dataset.pkl', 
        'train_dataset.pkl', 
        'val_dataset.pkl', 
        'best_model_smart.pt'
    ]
    
    for file in files_to_remove:
        path = Path('data') / file
        if path.exists():
            path.unlink()
            print(f"   ✓ Eliminado: {path}")
    
    # Carpetas a limpiar (solo contenido, no las carpetas)
    folders_to_clean = ['models', 'plots', 'results']
    
    for folder in folders_to_clean:
        folder_path = Path(folder)
        if folder_path.exists() and folder_path.is_dir():
            # Eliminar solo los archivos dentro de la carpeta, no la carpeta misma
            for item in folder_path.iterdir():
                if item.is_file():
                    item.unlink()  # Eliminar archivo
                    print(f"   ✓ Eliminado: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)  # Eliminar subcarpetas
                    print(f"   ✓ Eliminada subcarpeta: {item}")
            print(f"   ✓ Contenido limpiado: {folder_path}")

if __name__ == "__main__":
    clean_old_files()