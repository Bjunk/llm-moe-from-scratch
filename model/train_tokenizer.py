"""
Entrena un tokenizador BPE byte-level para código React.

Por qué byte-level:
- El tokenizador anterior usaba WhitespaceSplit, que no sabe cuándo hay espacio
  dentro de una palabra vs entre palabras → decodificaba setCount como "set Count".
- Byte-level codifica los espacios como parte de los tokens (igual que GPT-2),
  así que setCount, handleClick, useEffect se decodifican sin espacios espurios.

Uso:
    python3.10 train_tokenizer.py

Salida:
    bpe_react_tokenizer_v2.json  (úsalo en mini_coder2.py)
"""

import time
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ========================== CONFIGURACIÓN ==========================
ARCHIVO_DATASET   = "dataset/dataset_completo.txt"
ARCHIVO_SALIDA    = "bpe_react_tokenizer_v2.json"
VOCAB_SIZE        = 32000
MIN_FREQUENCY     = 2          # ignora tokens que aparecen menos de 2 veces

# ========================== ENTRENAMIENTO ==========================
print("🔧 Construyendo tokenizador byte-level BPE...")
tok = Tokenizer(BPE())
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|endoftext|>"],   # separador de documentos
    initial_alphabet=ByteLevel.alphabet(),  # los 256 bytes como base
    min_frequency=MIN_FREQUENCY,
    show_progress=True,
)

print(f"📚 Entrenando sobre {ARCHIVO_DATASET}...")
t0 = time.time()
tok.train(files=[ARCHIVO_DATASET], trainer=trainer)
print(f"✅ Entrenamiento completado en {(time.time()-t0)/60:.1f} min")
print(f"   Vocab real: {tok.get_vocab_size():,} tokens")

# ========================== VERIFICACIÓN ==========================
casos_camelcase = [
    "setCount", "handleClick", "useEffect", "setIsLoading",
    "fetchData", "useState", "useCallback", "className",
    "onClick", "onChange", "defaultValue", "AsyncThunk",
]
print("\n🔍 Verificación camelCase (debe decodificar sin espacios):")
ok = True
for w in casos_camelcase:
    dec = tok.decode(tok.encode(w).ids)
    estado = "✅" if dec == w else "❌"
    print(f"  {estado} {w:20s} → {repr(dec)}")
    if dec != w:
        ok = False

muestra = "const [count, setCount] = useState(0);"
dec_muestra = tok.decode(tok.encode(muestra).ids)
print(f"\n🔍 Frase completa:")
print(f"  original : {repr(muestra)}")
print(f"  decode   : {repr(dec_muestra)}")
print(f"  match    : {'✅' if muestra == dec_muestra else '❌'}")

# ========================== GUARDAR ================================
tok.save(ARCHIVO_SALIDA)
print(f"\n💾 Tokenizador guardado en: {ARCHIVO_SALIDA}")
print(f"   Actualiza ARCHIVO_TOKENIZER en mini_coder2.py a '{ARCHIVO_SALIDA}'")
