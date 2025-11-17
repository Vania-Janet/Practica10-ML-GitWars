#!/usr/bin/env python3
"""
Script de verificación antes de construir el contenedor Docker.
Valida que todos los archivos necesarios existen.
"""
import sys
from pathlib import Path

def verify():
    """Verifica que todo esté listo para construir."""
    errors = []
    warnings = []

    print("Verificando configuración del proyecto...\n")

    # Verificar estructura de directorios
    required_dirs = [
        "api",
        "api/model",
        "deployments"
    ]

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            errors.append(f"❌ Directorio faltante: {dir_path}")
        else:
            print(f"✓ Directorio encontrado: {dir_path}")

    # Verificar archivos críticos
    required_files = [
        "api/main.py",
        "api/requirements.txt",
        "api/model/model.pkl",
        "deployments/Dockerfile"
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            errors.append(f"❌ Archivo faltante: {file_path}")
        else:
            print(f"✓ Archivo encontrado: {file_path}")

    # Verificar que el modelo no esté vacío
    model_path = Path("api/model/model.pkl")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.1:
            warnings.append(f"⚠️  Modelo muy pequeño ({size_mb:.2f} MB)")
        else:
            print(f"✓ Modelo tiene tamaño adecuado: {size_mb:.2f} MB")

    # Verificar Dockerfile
    dockerfile = Path("deployments/Dockerfile")
    if dockerfile.exists():
        content = dockerfile.read_text()
        if "FROM python" not in content:
            errors.append("❌ Dockerfile no tiene imagen base de Python")
        if "uvicorn" not in content:
            warnings.append("⚠️  Dockerfile no menciona uvicorn")

    print()

    # Mostrar warnings
    if warnings:
        print("ADVERTENCIAS:")
        for warning in warnings:
            print(f"  {warning}")
        print()

    # Mostrar errores
    if errors:
        print("ERRORES CRÍTICOS:")
        for error in errors:
            print(f"  {error}")
        print("\n❌ Verificación fallida. Corrige los errores antes de construir.")
        return 1

    print("✅ Verificación exitosa. Listo para construir la imagen Docker.\n")
    return 0

if __name__ == "__main__":
    sys.exit(verify())
