import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings

# Servidor API
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Tokenización
from tokenizers import Tokenizer

warnings.filterwarnings("ignore", category=UserWarning)

# ============================ CONFIGURACIÓN ============================
ARCHIVO_DATASET   = "dataset/dataset_completo.txt"
ARCHIVO_MODELO    = "mini_moe_react_weights.pt"
ARCHIVO_TOKENIZER = "bpe_react_tokenizer_v2.json"   # byte-level BPE (sin camelCase roto)

LONGITUD_CONTEXTO  = 256           # 128 → 256: mejor coherencia en generación
DIM_EMBEDDING      = 2048
NUM_CABEZAS        = 8
DIM_FEEDFORWARD    = 512
NUM_CAPAS          = 6
NUM_EXPERTOS       = 8
TOP_K              = 2             # expertos activos por token (sparsity real)
TAMANO_VOCABULARIO = 32000         # debe coincidir con tok.get_vocab_size()

# --- Entrenamiento ---
LR            = 3e-4
EPOCAS        = 5
BATCH_SIZE    = 32                 # batch físico (cabe en MPS)
ACCUM_STEPS   = 8                  # batch efectivo = BATCH_SIZE * ACCUM_STEPS = 256
STRIDE        = 128                 # solape de ventanas (2x cobertura con 24M tokens)
VAL_FRAC      = 0.05               # fracción del corpus reservada a validación
AUX_LOSS_COEF = 0.01               # peso del load-balancing loss del MoE
GRAD_CLIP     = 1.0
DROPOUT       = 0.1

device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🍏 Hardware seleccionado para cómputo neuronal: {device}")


# ============================ ARQUITECTURA MoE ============================
class ExpertoReact(nn.Module):
    """Un FFN clásico. Varios de estos forman la capa MoE."""
    def __init__(self, dim_model, dim_feedforward, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(dim_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class CapaMoE(nn.Module):
    """
    MoE sparse top-k: cada token se enruta a TOP_K expertos.
    Devuelve la salida y un aux_loss (Switch Transformer) que balancea
    la carga entre expertos para evitar el colapso del router.
    """
    def __init__(self, dim_model, dim_feedforward, num_expertos=NUM_EXPERTOS, top_k=TOP_K):
        super().__init__()
        self.num_expertos = num_expertos
        self.top_k = top_k
        self.expertos = nn.ModuleList(
            [ExpertoReact(dim_model, dim_feedforward) for _ in range(num_expertos)]
        )
        self.router = nn.Linear(dim_model, num_expertos)

    def forward(self, x):
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)                       # (N, d)  N = b*t
        logits = self.router(x_flat)                    # (N, E)
        probs = F.softmax(logits, dim=-1)               # (N, E)

        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)        # (N, k)
        top_val = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-9)  # renormaliza
        # gate disperso (0 para expertos no elegidos): evita indexado booleano -> MPS-friendly
        gate = torch.zeros_like(probs).scatter_(1, top_idx, top_val)    # (N, E)

        salida = torch.zeros_like(x_flat)
        for e in range(self.num_expertos):
            # cada experto procesa TODOS los tokens (matmul denso) ponderado por su gate
            salida = salida + self.expertos[e](x_flat) * gate[:, e:e+1]

        # aux loss: f_i = fracción de tokens al experto i ; P_i = prob media del router
        with torch.no_grad():
            uno_hot = F.one_hot(top_idx[:, 0], self.num_expertos).float()
        f = uno_hot.mean(dim=0)                         # (E,)
        P = probs.mean(dim=0)                           # (E,)
        aux_loss = self.num_expertos * torch.sum(f * P)

        return salida.reshape(b, t, d), aux_loss


class BloqueTransformerMoE(nn.Module):
    """Bloque pre-norm: self-attention causal + capa MoE en lugar del FFN denso."""
    def __init__(self, dim, cabezas, dim_ff, num_expertos, top_k, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, cabezas, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = CapaMoE(dim, dim_ff, num_expertos, top_k)

    def forward(self, x, attn_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m, aux = self.moe(self.norm2(x))
        x = x + m
        return x, aux


class MiniMoECoder(nn.Module):
    def __init__(self, tamano_vocab, dim_embedding, num_cabezas,
                 dim_feedforward, num_capas, num_expertos, top_k=TOP_K):
        super().__init__()
        self.embedding = nn.Embedding(tamano_vocab, dim_embedding)
        self.pos_encoder = nn.Parameter(torch.randn(1, LONGITUD_CONTEXTO, dim_embedding) * 0.02)
        self.drop = nn.Dropout(DROPOUT)
        self.capas = nn.ModuleList([
            BloqueTransformerMoE(dim_embedding, num_cabezas, dim_feedforward, num_expertos, top_k)
            for _ in range(num_capas)
        ])
        self.norm_final = nn.LayerNorm(dim_embedding)
        self.fc = nn.Linear(dim_embedding, tamano_vocab, bias=False)
        self.fc.weight = self.embedding.weight          # weight tying
        self.apply(self._init_weights)                  # init estable (std 0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        t = x.size(1)
        h = self.embedding(x) + self.pos_encoder[:, :t, :]
        h = self.drop(h)
        mask = torch.triu(torch.full((t, t), -1e9, device=x.device), diagonal=1)
        aux_total = 0.0
        for capa in self.capas:
            h, aux = capa(h, mask)
            aux_total = aux_total + aux
        h = self.norm_final(h)
        return self.fc(h), aux_total / len(self.capas)  # logits: (b, t, vocab)


def construir_modelo():
    return MiniMoECoder(TAMANO_VOCABULARIO, DIM_EMBEDDING, NUM_CABEZAS,
                        DIM_FEEDFORWARD, NUM_CAPAS, NUM_EXPERTOS, TOP_K).to(device)


# ============================ DATOS ============================
def cargar_tokens():
    tokenizer = Tokenizer.from_file(ARCHIVO_TOKENIZER)
    assert tokenizer.get_vocab_size() == TAMANO_VOCABULARIO, (
        f"Vocab del tokenizer ({tokenizer.get_vocab_size()}) != TAMANO_VOCABULARIO ({TAMANO_VOCABULARIO})"
    )
    with open(ARCHIVO_DATASET, "r", encoding="utf-8") as f:
        ids = tokenizer.encode(f.read()).ids
    return np.array(ids, dtype=np.int64)


# ============================ ENTRENAMIENTO ============================
def entrenar(tokens_np=None):
    if tokens_np is None:
        tokens_np = cargar_tokens()

    modelo = construir_modelo()
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"🧠 Parámetros totales: {n_params/1e6:.1f}M | dataset: {len(tokens_np):,} tokens")

    # split por posición (no por ventana): nada de contexto se filtra entre train y val
    corte = int(len(tokens_np) * (1 - VAL_FRAC))
    idx_train = np.arange(0, corte - LONGITUD_CONTEXTO - 1, STRIDE, dtype=np.int64)
    idx_val   = np.arange(corte, len(tokens_np) - LONGITUD_CONTEXTO - 1, STRIDE, dtype=np.int64)
    desvios   = np.arange(LONGITUD_CONTEXTO, dtype=np.int64)
    print(f"   ventanas train: {len(idx_train):,} | val: {len(idx_val):,}")

    criterio = nn.CrossEntropyLoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)

    lotes_por_epoca = len(idx_train) // BATCH_SIZE
    pasos_opt_por_epoca = lotes_por_epoca // ACCUM_STEPS
    total_pasos = max(pasos_opt_por_epoca * EPOCAS, 1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizador, max_lr=LR,
                                              total_steps=total_pasos, pct_start=0.05)

    @torch.no_grad()
    def evaluar(max_lotes=40):
        modelo.eval()
        tot, n = 0.0, 0
        for nb in range(min(max_lotes, len(idx_val) // BATCH_SIZE)):
            idx = idx_val[nb*BATCH_SIZE:(nb+1)*BATCH_SIZE]
            bx = torch.from_numpy(tokens_np[idx[:, None] + desvios]).to(device)
            by = torch.from_numpy(tokens_np[idx[:, None] + desvios + 1]).to(device)
            lg, _ = modelo(bx)
            tot += criterio(lg.reshape(-1, TAMANO_VOCABULARIO), by.reshape(-1)).item()
            n += 1
        modelo.train()
        return tot / max(n, 1)

    mejor_val = float("inf")
    for epoca in range(EPOCAS):
        np.random.shuffle(idx_train)
        modelo.train()
        optimizador.zero_grad(set_to_none=True)

        for nb in range(lotes_por_epoca):
            idx = idx_train[nb*BATCH_SIZE:(nb+1)*BATCH_SIZE]
            x_np = tokens_np[idx[:, None] + desvios]
            y_np = tokens_np[idx[:, None] + desvios + 1]
            bx = torch.from_numpy(x_np).to(device)
            by = torch.from_numpy(y_np).to(device)

            logits, aux = modelo(bx)                    # (b, t, vocab)
            loss_ce = criterio(logits.reshape(-1, TAMANO_VOCABULARIO), by.reshape(-1))
            loss = (loss_ce + AUX_LOSS_COEF * aux) / ACCUM_STEPS
            loss.backward()

            if (nb + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
                optimizador.step()
                scheduler.step()
                optimizador.zero_grad(set_to_none=True)

            if (nb + 1) % 500 == 0:
                print(f"   ↳ [Época {epoca+1}] Lote {nb+1}/{lotes_por_epoca} -> "
                      f"CE: {loss_ce.item():.4f} | aux: {aux.item():.4f} | "
                      f"ppl: {math.exp(min(loss_ce.item(), 20)):.1f}")

        val = evaluar()
        print(f"📊 Época {epoca+1} -> val CE: {val:.4f} | val ppl: {math.exp(min(val, 20)):.1f}"
              + ("  ✅ (mejor)" if val < mejor_val else ""))
        if val < mejor_val:                             # guarda solo el mejor checkpoint
            mejor_val = val
            torch.save({"state_dict": modelo.state_dict(),
                        "optimizer": optimizador.state_dict(),
                        "epoca": epoca, "val": val}, ARCHIVO_MODELO)

    # recarga el mejor estado antes de devolver
    ckpt = torch.load(ARCHIVO_MODELO, map_location=device)
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo


# ============================ INFERENCIA ============================
@torch.no_grad()
def generar(modelo, tokenizer, prompt, max_tokens=64, temp=0.7, top_k=40, rep_penalty=1.3):
    """
    Genera texto a partir de un prompt.
    rep_penalty: penaliza tokens ya usados recientemente para evitar loops (1.0 = sin penalización).
    """
    modelo.eval()
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_tokens):
        ctx = torch.tensor([ids[-LONGITUD_CONTEXTO:]], dtype=torch.long, device=device)
        logits, _ = modelo(ctx)
        logits = logits[0, -1] / temp
        if rep_penalty != 1.0:
            for token_id in set(ids[-64:]):
                logits[token_id] = logits[token_id] / rep_penalty
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        prob = F.softmax(logits, dim=-1)
        ids.append(torch.multinomial(prob, 1).item())
    return tokenizer.decode(ids)


# ============================ API ============================
def crear_app(modelo, tokenizer):
    app = FastAPI()

    class PeticionCompletion(BaseModel):
        prompt: str
        max_tokens: int = 64
        temperature: float = 0.8

    @app.post("/v1/completions")
    def api_autocomplete(req: PeticionCompletion):
        texto = generar(modelo, tokenizer, req.prompt, req.max_tokens, req.temperature)
        return {"choices": [{"text": texto[len(req.prompt):]}]}

    return app


if __name__ == "__main__":
    if not os.path.exists(ARCHIVO_MODELO):
        print("📈 Iniciando entrenamiento...")
        modelo = entrenar()
    else:
        modelo = construir_modelo()
        ckpt = torch.load(ARCHIVO_MODELO, map_location=device)
        modelo.load_state_dict(ckpt["state_dict"])
        modelo.eval()
        print(f"✅ Modelo cargado (val ppl previa: {math.exp(min(ckpt.get('val', 0), 20)):.1f})")

    tokenizer = Tokenizer.from_file(ARCHIVO_TOKENIZER)
    uvicorn.run(crear_app(modelo, tokenizer), host="127.0.0.1", port=8000)
