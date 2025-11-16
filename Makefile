# Makefile para automatizar la construcción y ejecución del contenedor Docker
# Elemento 4 - API + Contenedor Docker + Makefile

# Variables
IMAGE_NAME = superhero-predictor
CONTAINER_NAME = superhero-api
PORT = 8000
TEAM_NAME = milanesa

# Colores para output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help build run status stop clean package train test all verify evaluate

# Target por defecto
.DEFAULT_GOAL := help

help: ## Muestra esta ayuda
	@echo "$(GREEN)Makefile para Superhero Power Predictor API$(NC)"
	@echo "$(YELLOW)Uso: make [target]$(NC)\n"
	@echo "Targets disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

train: ## Entrena el modelo y lo guarda en api/model/
	@echo "$(YELLOW)Entrenando modelo...$(NC)"
	python3 api/train_model.py
	@echo "$(GREEN)✓ Modelo entrenado y guardado$(NC)"

verify: ## Verifica que todo está listo para construir
	@echo "$(YELLOW)Verificando configuración...$(NC)"
	@python3 api/verify_build.py

build: verify ## Construye la imagen Docker
	@echo "$(YELLOW)Construyendo imagen Docker...$(NC)"
	docker build -f deployments/Dockerfile -t $(IMAGE_NAME):latest .
	@echo "$(GREEN)✓ Imagen construida: $(IMAGE_NAME):latest$(NC)"

run: ## Levanta el contenedor en segundo plano
	@echo "$(YELLOW)Iniciando contenedor...$(NC)"
	@if [ ! "$$(docker images -q $(IMAGE_NAME):latest)" ]; then \
		echo "$(RED)ERROR: Imagen no encontrada. Ejecuta 'make build' primero.$(NC)"; \
		exit 1; \
	fi
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "$(YELLOW)Contenedor ya está corriendo. Deteniéndolo primero...$(NC)"; \
		$(MAKE) stop; \
	fi
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		$(IMAGE_NAME):latest
	@echo "$(GREEN)✓ Contenedor iniciado: $(CONTAINER_NAME)$(NC)"
	@echo "$(YELLOW)Esperando que la API esté lista...$(NC)"
	@sleep 3
	@echo "$(GREEN)✓ API disponible en: http://localhost:$(PORT)$(NC)"
	@echo "$(YELLOW)Prueba: curl http://localhost:$(PORT)/health$(NC)"

status: ## Muestra el estado del contenedor
	@echo "$(YELLOW)Estado del contenedor:$(NC)"
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "$(GREEN)✓ Contenedor corriendo$(NC)"; \
		docker ps -f name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
		echo "\n$(YELLOW)Logs recientes:$(NC)"; \
		docker logs --tail 20 $(CONTAINER_NAME); \
	else \
		if [ "$$(docker ps -aq -f name=$(CONTAINER_NAME))" ]; then \
			echo "$(RED)✗ Contenedor existe pero está detenido$(NC)"; \
			docker ps -a -f name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}"; \
		else \
			echo "$(RED)✗ Contenedor no existe$(NC)"; \
		fi \
	fi

logs: ## Muestra los logs del contenedor
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker logs -f $(CONTAINER_NAME); \
	else \
		echo "$(RED)ERROR: Contenedor no está corriendo$(NC)"; \
		exit 1; \
	fi

stop: ## Detiene y elimina el contenedor
	@echo "$(YELLOW)Deteniendo contenedor...$(NC)"
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker stop $(CONTAINER_NAME); \
		echo "$(GREEN)✓ Contenedor detenido$(NC)"; \
	else \
		echo "$(YELLOW)Contenedor no está corriendo$(NC)"; \
	fi
	@if [ "$$(docker ps -aq -f name=$(CONTAINER_NAME))" ]; then \
		docker rm $(CONTAINER_NAME); \
		echo "$(GREEN)✓ Contenedor eliminado$(NC)"; \
	fi

clean: ## Limpia recursos de Docker (contenedor e imagen)
	@echo "$(YELLOW)Limpiando recursos Docker...$(NC)"
	@$(MAKE) stop || true
	@if [ "$$(docker images -q $(IMAGE_NAME):latest)" ]; then \
		docker rmi $(IMAGE_NAME):latest; \
		echo "$(GREEN)✓ Imagen eliminada$(NC)"; \
	else \
		echo "$(YELLOW)Imagen no encontrada$(NC)"; \
	fi
	@echo "$(GREEN)✓ Limpieza completada$(NC)"

package: ## Genera equipo_<nombre>.tar.gz
	@echo "$(YELLOW)Generando paquete para entrega...$(NC)"
	@if [ ! -f "api/model/model.pkl" ]; then \
		echo "$(RED)ERROR: Modelo no encontrado. Ejecuta 'make train' primero.$(NC)"; \
		exit 1; \
	fi
	@# Crear directorio temporal
	@rm -rf /tmp/gitwars-superheroes
	@mkdir -p /tmp/gitwars-superheroes
	@# Copiar archivos necesarios (excluyendo .git, __pycache__, etc.)
	@rsync -av --progress \
		--exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.DS_Store' \
		--exclude='*.tar.gz' \
		--exclude='.gitignore' \
		--exclude='notebooks' \
		. /tmp/gitwars-superheroes/
	@# Crear tarball
	@cd /tmp && tar -czf equipo_$(TEAM_NAME).tar.gz gitwars-superheroes/
	@mv /tmp/equipo_$(TEAM_NAME).tar.gz .
	@rm -rf /tmp/gitwars-superheroes
	@echo "$(GREEN)✓ Paquete creado: equipo_$(TEAM_NAME).tar.gz$(NC)"
	@echo "$(YELLOW)Contenido:$(NC)"
	@tar -tzf equipo_$(TEAM_NAME).tar.gz | head -20
	@echo "..."

test: ## Ejecuta tests de los endpoints
	@echo "$(YELLOW)Probando endpoints...$(NC)"
	@if [ ! "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "$(RED)ERROR: Contenedor no está corriendo. Ejecuta 'make run' primero.$(NC)"; \
		exit 1; \
	fi
	@echo "\n$(GREEN)1. Test /health$(NC)"
	@curl -s http://localhost:$(PORT)/health | python3 -m json.tool
	@echo "\n$(GREEN)2. Test /info$(NC)"
	@curl -s http://localhost:$(PORT)/info | python3 -m json.tool
	@echo "\n$(GREEN)3. Test /predict$(NC)"
	@curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"features": {"intelligence": 50, "strength": 80, "speed": 60, "durability": 70, "combat": 55, "height_cm": 185, "weight_kg": 90}}' \
		| python3 -m json.tool
	@echo "\n$(GREEN)✓ Todos los tests completados$(NC)"

all: train build run test ## Ejecuta todo el flujo: train, build, run, test
	@echo "$(GREEN)✓ Flujo completo ejecutado exitosamente$(NC)"

# Target para verificar el flujo del docente
evaluate: ## Simula el flujo EXACTO de evaluación del docente
	@echo "$(YELLOW)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(YELLOW)║    SIMULANDO FLUJO DE EVALUACIÓN DEL DOCENTE              ║$(NC)"
	@echo "$(YELLOW)╚════════════════════════════════════════════════════════════╝$(NC)\n"
	@echo "$(YELLOW)Paso 1: make build$(NC)"
	@$(MAKE) build
	@echo "\n$(YELLOW)Paso 2: make run$(NC)"
	@$(MAKE) run
	@sleep 5
	@echo "\n$(YELLOW)Paso 3: Probar endpoints con curl$(NC)"
	@echo "\n$(GREEN)▶ curl http://localhost:$(PORT)/health$(NC)"
	@curl -s http://localhost:$(PORT)/health | python3 -m json.tool
	@echo "\n$(GREEN)▶ curl http://localhost:$(PORT)/info$(NC)"
	@curl -s http://localhost:$(PORT)/info | python3 -m json.tool
	@echo "\n$(GREEN)▶ curl POST http://localhost:$(PORT)/predict$(NC)"
	@curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"features": {"intelligence": 50, "strength": 80, "speed": 60, "durability": 70, "combat": 55, "height_cm": 185, "weight_kg": 90}}' \
		| python3 -m json.tool
	@echo "\n$(YELLOW)Paso 4: make stop$(NC)"
	@$(MAKE) stop
	@echo "\n$(GREEN)✓ EVALUACIÓN COMPLETADA EXITOSAMENTE$(NC)"
	@echo "$(GREEN)  Todos los pasos del docente funcionan correctamente$(NC)\n"
