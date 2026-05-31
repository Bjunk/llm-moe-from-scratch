#!/bin/bash
# =============================================================================
# LLM MoE from Scratch — Pipeline completo
# github.com/Bjunk/llm-moe-from-scratch
#
# Estructura del proyecto:
#   scrapers/
#     extractor_github_v5.py   → código JSX/TSX desde GitHub (1000 repos)
#     scraper_docs.py          → docs oficiales (React, Next.js, TS, Tailwind...)
#   model/
#     train_tokenizer.py       → tokenizador byte-level BPE
#     mini_coder2.py           → entrenamiento del modelo MoE
#   test_model.py              → inferencia y pruebas
#   dataset/
#     mega_dataset_react.txt   → código scrapeado
#     docs_dataset.txt         → documentación oficial
#     dataset_completo.txt     → dataset unificado (input del modelo)
#
# Uso:
#   chmod +x start.sh
#   GITHUB_TOKEN=ghp_xxx ./start.sh
# =============================================================================

set -euo pipefail

# ── Colores ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}";
          echo -e "${BLUE}  $1${NC}";
          echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
ok()    { echo -e "${GREEN}  ✅ $1${NC}"; }
warn()  { echo -e "${YELLOW}  ⚠️  $1${NC}"; }
info()  { echo -e "${CYAN}  ℹ️  $1${NC}"; }
fatal() { echo -e "${RED}  ❌ $1${NC}"; exit 1; }
size()  { du -sh "$1" 2>/dev/null | cut -f1; }

# ── Detectar Python ────────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[ -z "$PYTHON" ] && fatal "Python no encontrado. Instala Python 3.10+."
ok "Python: $PYTHON ($($PYTHON --version 2>&1))"

# ── Verificar GITHUB_TOKEN ─────────────────────────────────────────────────
if [ -z "${GITHUB_TOKEN:-}" ]; then
    warn "GITHUB_TOKEN no definido — los scrapers funcionarán con rate limit bajo."
    warn "Para acceso completo: GITHUB_TOKEN=ghp_xxx ./start.sh"
    echo ""
else
    ok "GITHUB_TOKEN detectado"
fi

# ── Crear carpetas si no existen ───────────────────────────────────────────
mkdir -p dataset scrapers model

# ── Instalar dependencias ──────────────────────────────────────────────────
step "PASO 0 — Dependencias"
$PYTHON -m pip install --quiet \
    torch tokenizers fastapi uvicorn pydantic \
    requests gitpython
ok "Dependencias instaladas"

# =============================================================================
# PASO 1 — Scraping de código React desde GitHub
# =============================================================================
step "PASO 1 — Scraping de código React/Next.js (1000 repos)"
info "Script : scrapers/extractor_github_v5.py"
info "Salida : dataset/mega_dataset_react.txt"
info "Tiempo : ~6-10 horas (clonar 1000 repos)"

if [ -f "dataset/mega_dataset_react.txt" ]; then
    warn "mega_dataset_react.txt ya existe ($(size dataset/mega_dataset_react.txt)). Saltando."
    warn "Borra dataset/mega_dataset_react.txt para volver a scrapear."
else
    GITHUB_TOKEN="${GITHUB_TOKEN:-}" $PYTHON -u scrapers/extractor_github_v5.py
    # mover al directorio dataset si el script lo genera en la raíz
    [ -f "mega_dataset_react.txt" ] && mv mega_dataset_react.txt dataset/
    ok "Código scrapeado: $(size dataset/mega_dataset_react.txt)"
fi

# =============================================================================
# PASO 2 — Scraping de documentación oficial
# =============================================================================
step "PASO 2 — Scraping de documentación oficial"
info "Script : scrapers/scraper_docs.py"
info "Fuentes: React, Next.js, TypeScript, Tailwind, Radix UI, Zustand, TanStack..."
info "Salida : dataset/docs_dataset.txt"
info "Tiempo : ~2-5 minutos"

if [ -f "dataset/docs_dataset.txt" ]; then
    warn "docs_dataset.txt ya existe ($(size dataset/docs_dataset.txt)). Saltando."
    warn "Borra dataset/docs_dataset.txt para volver a scrapear."
else
    GITHUB_TOKEN="${GITHUB_TOKEN:-}" $PYTHON -u scrapers/scraper_docs.py
    [ -f "docs_dataset.txt" ] && mv docs_dataset.txt dataset/
    ok "Documentación scrapeada: $(size dataset/docs_dataset.txt)"
fi

# =============================================================================
# PASO 3 — Unir datasets en un único archivo
# =============================================================================
step "PASO 3 — Unificando datasets"
info "Entrada: dataset/mega_dataset_react.txt + dataset/docs_dataset.txt"
info "Salida : dataset/dataset_completo.txt"

[ ! -f "dataset/mega_dataset_react.txt" ] && fatal "mega_dataset_react.txt no existe. Ejecuta el PASO 1 primero."
[ ! -f "dataset/docs_dataset.txt"       ] && fatal "docs_dataset.txt no existe. Ejecuta el PASO 2 primero."

cat dataset/mega_dataset_react.txt dataset/docs_dataset.txt > dataset/dataset_completo.txt

TAMANO_CODIGO=$(size dataset/mega_dataset_react.txt)
TAMANO_DOCS=$(size dataset/docs_dataset.txt)
TAMANO_TOTAL=$(size dataset/dataset_completo.txt)

ok "Dataset unificado creado"
info "  Código   : $TAMANO_CODIGO"
info "  Docs     : $TAMANO_DOCS"
info "  Total    : $TAMANO_TOTAL  →  dataset/dataset_completo.txt"

# =============================================================================
# PASO 4 — Entrenar el tokenizador BPE byte-level
# =============================================================================
step "PASO 4 — Tokenizador byte-level BPE"
info "Script : model/train_tokenizer.py"
info "Salida : bpe_react_tokenizer_v2.json"
info "Tiempo : ~10-20 segundos"

if [ -f "bpe_react_tokenizer_v2.json" ]; then
    warn "bpe_react_tokenizer_v2.json ya existe. Saltando."
    warn "Bórralo si quieres re-entrenar el tokenizador."
else
    $PYTHON model/train_tokenizer.py
    ok "Tokenizador guardado: bpe_react_tokenizer_v2.json"
fi

# =============================================================================
# PASO 5 — Entrenar el modelo MoE
# =============================================================================
step "PASO 5 — Entrenamiento del modelo MoE (35M parámetros)"
info "Script : model/mini_coder2.py"
info "Arq.   : 6 capas Transformer + MoE (4 expertos, top-2)"
info "Dataset: dataset/dataset_completo.txt"
info "Hardware: MPS (Apple Silicon) / CUDA / CPU"
info "Tiempo : ~2-3 horas por época en Mac M4"
info "El mejor checkpoint se guarda automáticamente en mini_moe_react_weights.pt"

if [ -f "mini_moe_react_weights.pt" ]; then
    warn "mini_moe_react_weights.pt ya existe."
    echo -n "  ¿Entrenar de nuevo desde cero? [s/N]: "
    read -r respuesta
    if [[ "${respuesta:-N}" =~ ^[sS]$ ]]; then
        rm mini_moe_react_weights.pt
        $PYTHON -u model/mini_coder2.py
    else
        warn "Saltando entrenamiento. Usando checkpoint existente."
    fi
else
    $PYTHON -u model/mini_coder2.py
fi
ok "Modelo listo: mini_moe_react_weights.pt"

# =============================================================================
# PASO 6 — Probar el modelo
# =============================================================================
step "PASO 6 — Prueba de inferencia"
info "Script : test_model.py"
info "Genera completions React a partir de prompts reales."

$PYTHON test_model.py

# =============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Pipeline completo ✅${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Archivos generados:"
echo "    📁 dataset/"
echo "       mega_dataset_react.txt    — código JSX/TSX (1000 repos)"
echo "       docs_dataset.txt          — documentación oficial"
echo "       dataset_completo.txt      — dataset unificado"
echo "    🔤 bpe_react_tokenizer_v2.json"
echo "    🧠 mini_moe_react_weights.pt"
echo ""
echo "  Para experimentar:"
echo "    $PYTHON test_model.py                   — probar prompts"
echo "    $PYTHON -u model/mini_coder2.py         — continuar entrenamiento"
echo ""
echo "  github.com/Bjunk/llm-moe-from-scratch"
echo ""
