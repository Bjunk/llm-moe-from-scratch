from mini_coder2 import construir_modelo, generar, device
from tokenizers import Tokenizer
import torch

# ── Cargar modelo ──────────────────────────────────────────────────
modelo = construir_modelo()
ckpt = torch.load("mini_moe_react_weights.pt", map_location=device)
modelo.load_state_dict(ckpt["state_dict"])
modelo.eval()
print(f"✅ Modelo cargado\n")

# ── Cargar tokenizador ─────────────────────────────────────────────
tok = Tokenizer.from_file("bpe_react_tokenizer_v2.json")

# ── Prompts de prueba ──────────────────────────────────────────────
prompts = [
    "const [count, setCount] = useState(",
    "function Button({ onClick, children",
    "useEffect(() => {",
    "export default function App() {",
]

for p in prompts:
    print(f"🔹 {p}")
    resultado = generar(modelo, tok, p, max_tokens=60, temp=0.7, rep_penalty=1.3)
    # mostrar solo la parte generada, sin el prompt
    print(resultado[len(p):])
    print()
