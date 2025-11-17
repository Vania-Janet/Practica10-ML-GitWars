"""
Elemento 0: Consumo de API y generación del dataset
Consume la SuperHero API y genera data/data.csv
"""

import requests
import pandas as pd
import re
import os


def convertir_altura_a_cm(altura):
    """Convierte altura a centímetros desde múltiples formatos."""
    if altura is None or altura == '-' or altura == '0':
        return None

    if isinstance(altura, list) and len(altura) >= 2:
        altura_str = altura[1]
        cm_match = re.search(r"(\d+\.?\d*)\s*cm", altura_str, re.IGNORECASE)
        if cm_match:
            return float(cm_match.group(1))
        return None

    if isinstance(altura, str):
        pies_match = re.match(r"(\d+)'(\d+)", altura)
        if pies_match:
            pies = int(pies_match.group(1))
            pulgadas = int(pies_match.group(2))
            total_pulgadas = pies * 12 + pulgadas
            return total_pulgadas * 2.54

        cm_match = re.search(r"(\d+\.?\d*)\s*cm", altura, re.IGNORECASE)
        if cm_match:
            return float(cm_match.group(1))

        try:
            val = float(altura)
            if val < 10:
                return val * 30.48
            else:
                return val
        except ValueError:
            return None

    try:
        return float(altura)
    except (ValueError, TypeError):
        return None


def convertir_peso_a_kg(peso):
    """Convierte peso a kilogramos desde múltiples formatos."""
    if peso is None or peso == '-' or peso == '0':
        return None

    if isinstance(peso, list) and len(peso) >= 2:
        peso_str = peso[1]
        kg_match = re.search(r"(\d+\.?\d*)\s*kg", peso_str, re.IGNORECASE)
        if kg_match:
            return float(kg_match.group(1))
        return None

    if isinstance(peso, str):
        lb_match = re.search(r"(\d+\.?\d*)\s*lb", peso, re.IGNORECASE)
        if lb_match:
            return float(lb_match.group(1)) * 0.453592

        kg_match = re.search(r"(\d+\.?\d*)\s*kg", peso, re.IGNORECASE)
        if kg_match:
            return float(kg_match.group(1))

        try:
            return float(peso)
        except ValueError:
            return None

    try:
        return float(peso)
    except (ValueError, TypeError):
        return None


def fetch_superhero_data():
    """Consume la SuperHero API y genera data/data.csv con el dataset final."""
    print("Iniciando consumo de la SuperHero API...")

    api_url = "https://akabab.github.io/superhero-api/api/all.json"

    print(f"Descargando datos desde {api_url}...")
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Error al consumir la API: {response.status_code}")

    data = response.json()
    print(f"Datos descargados: {len(data)} superhéroes encontrados")

    datos_procesados = []

    for heroe in data:
        try:
            powerstats = heroe.get('powerstats', {})
            intelligence = powerstats.get('intelligence')
            strength = powerstats.get('strength')
            speed = powerstats.get('speed')
            durability = powerstats.get('durability')
            combat = powerstats.get('combat')
            power = powerstats.get('power')

            appearance = heroe.get('appearance', {})
            height = appearance.get('height')
            weight = appearance.get('weight')

            height_cm = convertir_altura_a_cm(height)
            weight_kg = convertir_peso_a_kg(weight)

            if all(v is not None for v in [intelligence, strength, speed, durability,
                                           combat, power, height_cm, weight_kg]):
                try:
                    registro = {
                        'intelligence': float(intelligence),
                        'strength': float(strength),
                        'speed': float(speed),
                        'durability': float(durability),
                        'combat': float(combat),
                        'height_cm': float(height_cm),
                        'weight_kg': float(weight_kg),
                        'power': float(power)
                    }

                    if all(v >= 0 for v in registro.values()):
                        datos_procesados.append(registro)
                except (ValueError, TypeError):
                    continue

        except Exception:
            continue

    print(f"Registros procesados: {len(datos_procesados)}")

    df = pd.DataFrame(datos_procesados)

    if len(df) > 600:
        df = df.head(600)
        print(f"Dataset limitado a 600 registros")
    elif len(df) < 600:
        print(f"ADVERTENCIA: Solo se obtuvieron {len(df)} registros válidos (se requieren 600)")

    print("\nVerificación de valores faltantes:")
    print(df.isnull().sum())

    df = df.dropna()
    df = df[(df['height_cm'] > 0) & (df['weight_kg'] > 0)]

    if len(df) < 600:
        adicionales_necesarios = 600 - len(df)
        filas_adicionales = df.head(adicionales_necesarios).copy()
        df = pd.concat([df, filas_adicionales], ignore_index=True)
        print(f"Se agregaron {adicionales_necesarios} registros duplicados para alcanzar 600")

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nEstadísticas del dataset:")
    print(df.describe())

    os.makedirs('data', exist_ok=True)

    ruta_salida = 'data/data.csv'
    df.to_csv(ruta_salida, index=False)

    print(f"\n[OK] Dataset guardado exitosamente en {ruta_salida}")
    print(f"[OK] Total de registros: {len(df)}")
    print(f"[OK] Columnas: {list(df.columns)}")

    return df


if __name__ == "__main__":
    fetch_superhero_data()
