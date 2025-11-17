"""
Script de validación para el dataset generado
Verifica que cumple todos los criterios del Elemento 0
"""

import pandas as pd
import os


def validate_dataset():
    """Valida que el dataset cumple con todos los requisitos del Elemento 0."""
    print("VALIDACIÓN DEL DATASET - ELEMENTO 0")
    print("-" * 60)

    file_path = 'data/data.csv'
    if not os.path.exists(file_path):
        print(f"ERROR: El archivo {file_path} no existe")
        return False

    print(f"[OK] Archivo {file_path} encontrado")

    df = pd.read_csv(file_path)

    num_registros = len(df)
    if num_registros == 600:
        print(f"✓ Número de registros: {num_registros} (CORRECTO)")
    else:
        print(f"ERROR: Número de registros: {num_registros} (SE REQUIEREN 600)")
        return False

    columnas_esperadas = ['intelligence', 'strength', 'speed', 'durability',
                          'combat', 'height_cm', 'weight_kg', 'power']

    if list(df.columns) == columnas_esperadas:
        print(f"✓ Columnas correctas: {list(df.columns)}")
    else:
        print(f"ERROR: Columnas incorrectas")
        print(f"   Esperadas: {columnas_esperadas}")
        print(f"   Obtenidas: {list(df.columns)}")
        return False

    valores_faltantes = df.isnull().sum().sum()
    if valores_faltantes == 0:
        print(f"✓ No hay valores faltantes")
    else:
        print(f"ERROR: Hay {valores_faltantes} valores faltantes:")
        print(df.isnull().sum())
        return False

    no_numericas = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            no_numericas.append(col)

    if len(no_numericas) == 0:
        print(f"✓ Todas las columnas son numéricas")
    else:
        print(f"ERROR: Columnas no numéricas: {no_numericas}")
        return False

    print("\nESTADÍSTICAS DEL DATASET")
    print("-" * 60)
    print(df.describe())

    print("\nVALIDACIÓN EXITOSA")
    print("-" * 60)
    print("El dataset cumple con todos los requisitos del Elemento 0:")
    print("  ✓ 600 registros")
    print("  ✓ 8 columnas correctas")
    print("  ✓ Sin valores faltantes")
    print("  ✓ Todas las columnas numéricas")

    return True


if __name__ == "__main__":
    validate_dataset()
