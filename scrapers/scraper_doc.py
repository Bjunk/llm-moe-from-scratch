"""
Scraper de documentación oficial para enriquecer el dataset de entrenamiento.

Por qué documentación + código:
  - El código enseña el CÓMO. La documentación enseña el CUÁNDO y el POR QUÉ.
  - El modelo aprende a asociar lenguaje natural con implementaciones concretas.
  - Las secciones "best practices" y "avoid" no aparecen en código — solo en docs.
  - Ratio recomendado: ~80% código + ~20% documentación.

Fuentes incluidas:
  ✅ Sin token (60 req/hr)        | Con GITHUB_TOKEN (5000 req/hr)
  ─────────────────────────────── | ────────────────────────────────
  Tailwind CSS (187 archivos)     | React Docs (react.dev)
  React Hook Form (16 archivos)   | Next.js Docs
  Radix UI Primitives             | shadcn/ui
  Jotai (state management)        | TypeScript Handbook
                                  | Zustand

Uso:
  # Sin token (fuentes públicas):
  python3.10 scraper_docs.py

  # Con token (todas las fuentes — recomendado):
  GITHUB_TOKEN=ghp_xxx python3.10 scraper_docs.py

Salida:
  docs_dataset.txt — listo para concatenar con mega_dataset_react.txt

  Para combinar ambos datasets antes de entrenar:
  cat mega_dataset_react.txt docs_dataset.txt > dataset_completo.txt
  # luego cambia ARCHIVO_DATASET = "dataset_completo.txt" en mini_coder2.py
"""

import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================== CONFIGURACIÓN ==========================
ARCHIVO_SALIDA    = "docs_dataset.txt"
WORKERS           = 10           # descargas paralelas
DELAY_API         = 0.2          # segundos entre llamadas a la Contents API
MIN_CHARS         = 150          # descartar docs demasiado cortos
SEPARADOR         = "\n\n<|endoftext|>\n\n"  # separador entre documentos

# Token desde variable de entorno (nunca hardcodeado)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
if GITHUB_TOKEN:
    print("🔑 GITHUB_TOKEN detectado — acceso a todas las fuentes")
else:
    print("⚠️  Sin GITHUB_TOKEN — usando solo fuentes públicas (60 req/hr)")
    print("   Para más fuentes: GITHUB_TOKEN=ghp_xxx python3.10 scraper_docs.py\n")

# ========================== FUENTES ================================
# Formato: (owner, repo, ruta_docs, branch, nombre_display, requiere_token)
FUENTES = [
    # ── Sin token ──────────────────────────────────────────────────
    (
        "tailwindlabs", "tailwindcss.com",
        "src/pages/docs", "master",
        "Tailwind CSS", False,
    ),
    (
        "react-hook-form", "react-hook-form",
        "docs", "master",
        "React Hook Form", False,
    ),
    (
        "radix-ui", "website",
        "data/primitives/docs", "main",
        "Radix UI", False,
    ),
    (
        "pmndrs", "zustand",
        "docs", "main",
        "Zustand", False,
    ),
    # ── Con GITHUB_TOKEN ───────────────────────────────────────────
    (
        "reactjs", "react.dev",
        "src/content", "main",
        "React Docs", True,
    ),
    (
        "vercel", "next.js",
        "docs", "canary",
        "Next.js Docs", True,
    ),
    (
        "shadcn-ui", "ui",
        "apps/www/content/docs", "main",
        "shadcn/ui", True,
    ),
    (
        "microsoft", "TypeScript-Website",
        "packages/documentation/copy/en", "v2",
        "TypeScript Handbook", True,
    ),
    (
        "pmndrs", "jotai",
        "docs", "main",
        "Jotai", True,
    ),
    (
        "TanStack", "query",
        "docs/framework/react", "main",
        "TanStack Query", True,
    ),
]


# ========================== LIMPIADOR MDX ==========================
def limpiar_mdx(texto: str, titulo: str = "", fuente: str = "") -> str:
    """
    Limpia archivos MD/MDX para entrenamiento:
    - Preserva: texto explicativo, bloques de código, headers
    - Elimina: frontmatter YAML, imports MDX, componentes JSX custom
    """
    # 0. Proteger bloques de código antes de cualquier limpieza
    bloques: dict[str, str] = {}
    contador = [0]

    def guardar(m: re.Match) -> str:
        key = f"__BLK{contador[0]}__"
        bloques[key] = m.group(0)
        contador[0] += 1
        return key

    texto = re.sub(r"```[\s\S]*?```", guardar, texto)   # bloques triple backtick
    texto = re.sub(r"`[^`\n]{1,200}`", guardar, texto)  # inline code

    # 1. Eliminar frontmatter YAML (---...---)
    texto = re.sub(r"^---[\s\S]*?---\s*", "", texto)

    # 2. Eliminar imports MDX (los que están fuera de code blocks, ya protegidos)
    texto = re.sub(r"^import\s+.*\n", "", texto, flags=re.MULTILINE)

    # 3. Eliminar componentes JSX personalizados (PascalCase) con su contenido
    #    Ej: <Note>...</Note>, <Callout>...</Callout>
    for _ in range(3):  # múltiples pasadas para anidamiento
        texto = re.sub(
            r"<([A-Z][A-Za-z]*)\b[^>]*>[\s\S]*?</\1>",
            "",
            texto,
        )

    # 4. Eliminar self-closing JSX custom: <Icon />, <Demo />
    texto = re.sub(r"<[A-Z][A-Za-z]*\b[^>]*/?>", "", texto)

    # 5. Eliminar atributos style y className sueltos (artefactos del parseo)
    texto = re.sub(r"\{\s*/\*.*?\*/\s*\}", "", texto)

    # 6. Restaurar bloques de código
    for key, bloque in bloques.items():
        texto = texto.replace(key, bloque)

    # 7. Limpiar líneas vacías excesivas
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = texto.strip()

    # 8. Descartar si quedó muy corto (era casi todo componentes)
    if len(texto) < MIN_CHARS:
        return ""

    # 9. Header de fuente — enseña al modelo el contexto del texto
    if titulo and fuente:
        texto = f"# [{fuente}] {titulo}\n\n{texto}"

    return texto


def titulo_desde_nombre(nombre_archivo: str) -> str:
    return (
        nombre_archivo
        .replace(".mdx", "")
        .replace(".md", "")
        .replace("-", " ")
        .replace("_", " ")
        .title()
    )


# ========================== DESCARGA ==============================
def descargar_archivo(download_url: str, titulo: str, fuente: str) -> str:
    """Descarga y limpia un archivo de documentación."""
    try:
        r = requests.get(download_url, timeout=20)
        r.raise_for_status()
        texto = limpiar_mdx(r.text, titulo, fuente)
        return texto
    except Exception:
        return ""


def listar_archivos_recursivo(
    owner: str, repo: str, ruta: str, branch: str, extensiones: tuple
) -> list[dict]:
    """
    Lista todos los archivos con las extensiones dadas en una ruta del repo,
    recursando en subdirectorios.
    """
    archivos = []
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{ruta}?ref={branch}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 403:
            if "rate limit" in r.text.lower():
                print(f"  ⏳ Rate limit alcanzado, esperando 60s...")
                time.sleep(60)
                r = requests.get(url, headers=HEADERS, timeout=15)
            else:
                return []  # requiere token que no tenemos
        if r.status_code != 200:
            return []

        for item in r.json():
            if not isinstance(item, dict):
                continue
            if item.get("type") == "file" and item["name"].endswith(extensiones):
                archivos.append(item)
            elif item.get("type") == "dir":
                time.sleep(DELAY_API)
                archivos.extend(
                    listar_archivos_recursivo(
                        owner, repo, item["path"], branch, extensiones
                    )
                )
    except Exception as e:
        print(f"  ⚠ Error listando {ruta}: {e}")
    return archivos


# ========================== MAIN ==================================
def main():
    todos_los_textos: list[str] = []
    stats = {"fuentes": 0, "archivos": 0, "chars": 0, "saltados": 0}

    for owner, repo, ruta, branch, nombre, req_token in FUENTES:
        if req_token and not GITHUB_TOKEN:
            print(f"⏩ {nombre} (requiere GITHUB_TOKEN — saltando)")
            continue

        print(f"\n📚 {nombre} ({owner}/{repo}/{ruta})")

        archivos = listar_archivos_recursivo(
            owner, repo, ruta, branch, (".md", ".mdx")
        )
        if not archivos:
            print(f"   ⚠ Sin archivos encontrados")
            continue

        print(f"   {len(archivos)} archivos — descargando en paralelo...")

        # Descarga paralela
        textos_fuente: list[str] = []
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            tareas = {
                executor.submit(
                    descargar_archivo,
                    item["download_url"],
                    titulo_desde_nombre(item["name"]),
                    nombre,
                ): item["name"]
                for item in archivos
                if "download_url" in item
            }
            for fut in as_completed(tareas):
                texto = fut.result()
                if texto:
                    textos_fuente.append(texto)
                else:
                    stats["saltados"] += 1

        chars_fuente = sum(len(t) for t in textos_fuente)
        tokens_est = chars_fuente // 4
        print(f"   ✅ {len(textos_fuente)} docs | {chars_fuente:,} chars | ~{tokens_est:,} tokens")

        todos_los_textos.extend(textos_fuente)
        stats["fuentes"] += 1
        stats["archivos"] += len(textos_fuente)
        stats["chars"] += chars_fuente

    if not todos_los_textos:
        print("\n❌ No se descargó ningún documento.")
        return

    # Guardar con separadores de documento
    print(f"\n💾 Guardando {ARCHIVO_SALIDA}...")
    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        f.write(SEPARADOR.join(todos_los_textos))

    mb = os.path.getsize(ARCHIVO_SALIDA) / (1024 * 1024)
    tokens_total = stats["chars"] // 4

    print(f"\n{'='*60}")
    print(f"✅ Dataset de documentación generado")
    print(f"   Fuentes procesadas : {stats['fuentes']}")
    print(f"   Documentos         : {stats['archivos']:,}")
    print(f"   Saltados/vacíos    : {stats['saltados']:,}")
    print(f"   Tamaño             : {mb:.1f} MB")
    print(f"   Tokens estimados   : ~{tokens_total:,}")
    print(f"{'='*60}")
    print(f"""
Para combinar con tu dataset de código:

  cat mega_dataset_react.txt docs_dataset.txt > dataset_completo.txt

Luego en mini_coder2.py cambia:
  ARCHIVO_DATASET = "dataset_completo.txt"

Ratio resultante: {stats['chars'] // 4:,} tokens docs sobre tu dataset total.
""")


if __name__ == "__main__":
    main()
