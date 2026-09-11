.PHONY: install install-system install-python install-models index run clean help

PYTHON := python3.12
VENV := venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

help:
	@echo "Targets disponibles:"
	@echo "  make install        - Instala todo (sistema + python + modelos)"
	@echo "  make install-system - Instala dependencias del sistema (apt)"
	@echo "  make install-python - Crea venv e instala paquetes de Python"
	@echo "  make install-models - Descarga los modelos de Ollama"
	@echo "  make index          - Indexa los PDFs de la carpeta documentos/"
	@echo "  make run            - Corre el asistente"
	@echo "  make clean          - Borra el entorno virtual"

install: install-system install-python install-models
	@echo ""
	@echo "Instalación completa."
	@echo "Coloca tus PDFs en documentos/ y corre 'make index', luego 'make run'."

install-system:
	@echo "Instalando dependencias del sistema..."
	sudo apt-get update
	sudo apt-get install -y $(PYTHON) $(PYTHON)-venv $(PYTHON)-dev build-essential git curl espeak-ng portaudio19-dev
	@echo "Instalando dependencias de Qt (interfaz gráfica del orbe)..."
	sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "Instalando Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	else \
		echo "Ollama ya está instalado."; \
	fi

install-python:
	@echo "Creando entorno virtual..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@echo "Instalando PyTorch con soporte CUDA..."
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu121
	@echo "Instalando el resto de dependencias..."
	$(PIP) install -r requirements.txt

install-models:
	@echo "Descargando modelos de Ollama (esto puede tardar)..."
	ollama pull gemma3:12b
	ollama pull nomic-embed-text

index:
	@mkdir -p documentos
	@if [ -z "$$(ls -A documentos 2>/dev/null)" ]; then \
		echo "No hay PDFs en documentos/. Coloca tus archivos ahí primero."; \
	else \
		echo "Indexando PDFs..."; \
		$(PY) indexar_pdfs.py; \
	fi

run:
	@echo "Si usas Linux con Wayland y el orbe no se comporta bien (no se puede"
	@echo "arrastrar, no queda encima de otras ventanas), corre en su lugar:"
	@echo "  QT_QPA_PLATFORM=xcb make run"
	$(PY) asistente.py

clean:
	rm -rf $(VENV)
	@echo "Entorno virtual eliminado."
