#!/bin/bash
# =============================================================================
# LLM MoE from Scratch — Pipeline completo
# github.com/Bjunk/llm-moe-from-scratch
#
# Uso:
#   chmod +x run.sh
#   ./run.sh
# =============================================================================

set -e  # detiene el script si cualquier paso falla

# ── Colores ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # sin color

step()  { echo -e "\n${BLUE}━━━ $1 ${NC}"; }
ok()    { echo -e "${GREEN}✅ $1${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $1${NC}"; }
fatal() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ── Detectar Python ────────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[ -z "$PYTHON" ] && fatal "No se encontró Python. Instala Python 3.10+."
ok "Python detectado: $PYTHON ($($PYTHON --version))"

# ── Instalar dependencias ──────────────────────────────────────────────────
step "PASO 0 — Instalando dependencias"
$PYTHON -m pip install --quiet torch tokenizers fastapi uvicorn pydantic requests
ok "Dependencias listas"

# =============================================================================
# PASO 1 — Scraping de repositorios GitHub
# =============================================================================
step "PASO 1 — Scraping de repositorios React desde GitHub"
echo "   Descarga código JSX/TSX real de repositorios públicos."
echo "   Resultado: mega_dataset_react.txt"
echo ""

if [ -f "mega_dataset_react.txt" ]; then
    warn "mega_dataset_react.txt ya existe. Saltando scraping."
    warn "Bórralo manualmente si quieres volver a scrapear."
else
    $PYTHON extractor_github_v4.py
    ok "Dataset generado: mega_dataset_react.txt"
fi

# =============================================================================
# PASO 2 — Entrenar el tokenizador BPE byte-level
# =============================================================================
step "PASO 2 — Entrenando tokenizador byte-level BPE"
echo "   Aprende a tokenizar código React sin romper camelCase."
echo "   Resultado: bpe_react_tokenizer_v2.json (~6 segundos)"
echo ""

if [ -f "bpe_react_tokenizer_v2.json" ]; then
    warn "bpe_react_tokenizer_v2.json ya existe. Saltando."
else
    $PYTHON train_tokenizer.py
    ok "Tokenizador guardado: bpe_react_tokenizer_v2.json"
fi

# =============================================================================
# PASO 3 — Entrenar el modelo MoE
# =============================================================================
step "PASO 3 — Entrenando el modelo MoE (35M parámetros)"
echo "   Arquitectura: 6 capas Transformer + Mixture of Experts (4 expertos, top-2)"
echo "   Hardware: MPS (Apple Silicon) / CUDA / CPU"
echo "   Duración estimada: 2-3 horas por época en Mac M4"
echo "   El mejor checkpoint se guarda automáticamente."
echo ""

if [ -f "mini_moe_react_weights.pt" ]; then
    warn "mini_moe_react_weights.pt ya existe."
    echo -n "   ¿Entrenar de nuevo y sobreescribir? [s/N]: "
    read -r respuesta
    if [[ "$respuesta" =~ ^[sS]$ ]]; then
        rm mini_moe_react_weights.pt
        $PYTHON -u mini_coder2.py
    else
        warn "Saltando entrenamiento. Usando checkpoint existente."
    fi
else
    $PYTHON -u mini_coder2.py
fi
ok "Modelo entrenado: mini_moe_react_weights.pt"

# =============================================================================
# PASO 4 — Probar el modelo
# =============================================================================
step "PASO 4 — Probando el modelo con prompts React"
echo "   Genera completions a partir de prompts reales."
echo ""

$PYTHON test_model.py

# =============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Pipeline completo ✅${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Archivos generados:"
echo "    📄 mega_dataset_react.txt       — dataset de entrenamiento"
echo "    🔤 bpe_react_tokenizer_v2.json  — tokenizador byte-level BPE"
echo "    🧠 mini_moe_react_weights.pt    — pesos del modelo entrenado"
echo ""
echo "  Para seguir experimentando:"
echo "    $PYTHON test_model.py           — probar más prompts"
echo "    $PYTHON -u mini_coder2.py       — continuar entrenamiento"
echo ""
echo "  github.com/Bjunk/llm-moe-from-scratch"
echo ""
