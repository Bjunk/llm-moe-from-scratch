"""
Extractor de código React/Next.js desde GitHub — v5
- Solo fuentes públicas de GitHub (sin código local)
- 1000 repositorios con paginación automática
- Token desde variable de entorno (nunca hardcodeado)
- ETA en tiempo real

Uso:
    GITHUB_TOKEN=ghp_tu_nuevo_token python3.10 extractor_github_v5.py
"""

import os
import re
import math
import shutil
import requests
import time
from git import Repo

# ========================== CONFIGURACIÓN ==========================
GITHUB_TOKEN     = os.environ.get("GITHUB_TOKEN", "")   # NUNCA hardcodeado
CANTIDAD_REPOS   = 1000
ESTRELLAS_MIN    = 100        # ≥100 stars garantiza calidad y alcanza 1000 repos
TEMPORAL_DIR     = "./temp_repos"
ARCHIVO_SALIDA   = "dataset/mega_dataset_react.txt"
CHECKPOINT_FILE  = "extractor_checkpoint.txt"  # guarda progreso para reanudar
MIN_CHARS_ARCHIVO = 50        # descartar archivos demasiado pequeños

RUIDO_CARPETAS = {
    "node_modules", ".next", "dist", "cypress", ".git",
    "build", "public", "out", "coverage", "scripts",
    ".turbo", ".vercel", "storybook-static",
}
RUIDO_ARCHIVOS = {
    "test", "spec", "webpack", "eslint", "jest",
    "changelog", "license", "mock", "fixture", "story",
}

if not GITHUB_TOKEN:
    print("⚠️  Sin GITHUB_TOKEN — rate limit bajo (10 req/min)")
    print("   Uso: GITHUB_TOKEN=ghp_xxx python3.10 extractor_github_v5.py\n")
else:
    print("🔑 GITHUB_TOKEN detectado — usando rate limit autenticado\n")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


# ========================== BUSQUEDA CON PAGINACION ================
def obtener_repositorios(cantidad: int, estrellas_min: int) -> list[str]:
    """
    Obtiene hasta `cantidad` repos paginando la API de búsqueda de GitHub.
    La API permite máx 100 por página y 10 páginas = 1000 resultados totales.
    """
    print(f"🔍 Buscando {cantidad} repositorios React/Next.js con ≥{estrellas_min} ⭐...")
    repos = []
    paginas = math.ceil(min(cantidad, 1000) / 100)

    for pagina in range(1, paginas + 1):
        url = (
            f"https://api.github.com/search/repositories"
            f"?q=topic:nextjs+topic:react+stars:>={estrellas_min}"
            f"&sort=stars&order=desc&per_page=100&page={pagina}"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                espera = max(reset - int(time.time()), 10)
                print(f"  ⏳ Rate limit — esperando {espera}s...")
                time.sleep(espera)
                r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  ❌ Error API página {pagina}: {r.status_code}")
                break
            items = r.json().get("items", [])
            if not items:
                break
            batch = [i["clone_url"] for i in items]
            repos.extend(batch)
            total_api = r.json().get("total_count", "?")
            print(f"  📄 Página {pagina}/{paginas} — {len(repos)}/{min(cantidad, total_api)} repos")
            time.sleep(1.5)  # respetar rate limit de búsqueda (30/min auth)
        except Exception as e:
            print(f"  ❌ Error en página {pagina}: {e}")
            break

    # Deduplicar por nombre de repo
    vistos = set()
    unicos = []
    for url in repos:
        nombre = url.split("/")[-1]
        if nombre not in vistos:
            vistos.add(nombre)
            unicos.append(url)

    print(f"✅ {len(unicos)} repos únicos encontrados\n")
    return unicos[:cantidad]


# ========================== PROCESAMIENTO ==========================
def limpiar_codigo(codigo: str) -> str:
    """Elimina comentarios y normaliza whitespace para reducir ruido."""
    codigo = re.sub(r"//.*", "", codigo)
    codigo = re.sub(r"/\*.*?\*/", "", codigo, flags=re.DOTALL)
    codigo = re.sub(r"[ \t]+", " ", codigo)
    codigo = re.sub(r"\n\s*\n", "\n", codigo)
    return codigo.strip()


def procesar_repositorio(url_repo: str, index: int, total: int, t_inicio: float) -> str:
    nombre = url_repo.split("/")[-1].replace(".git", "")
    ruta   = os.path.join(TEMPORAL_DIR, nombre)

    # ETA estimada
    elapsed = time.time() - t_inicio
    eta_min = (elapsed / max(index - 1, 1)) * (total - index + 1) / 60 if index > 1 else "?"
    eta_str = f"{eta_min:.0f} min" if isinstance(eta_min, float) else eta_min
    print(f"🌐 [{index}/{total}] {nombre:<40} ETA: ~{eta_str}")

    if os.path.exists(ruta):
        shutil.rmtree(ruta)

    try:
        Repo.clone_from(url_repo, ruta, depth=1, single_branch=True)
        texto = ""
        for raiz, dirs, archivos in os.walk(ruta):
            dirs[:] = [
                d for d in dirs
                if d.lower() not in RUIDO_CARPETAS and not d.startswith(".")
            ]
            for archivo in archivos:
                ruta_completa = os.path.join(raiz, archivo)

                # Archivos JSX/TSX
                if archivo.endswith((".jsx", ".tsx")):
                    if any(r in archivo.lower() for r in RUIDO_ARCHIVOS):
                        continue
                    try:
                        with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                            limpio = limpiar_codigo(f.read())
                        if len(limpio) >= MIN_CHARS_ARCHIVO:
                            texto += f"\n\n/* [{nombre}] {archivo} */\n{limpio}"
                    except Exception:
                        pass

                # Configuración de Tailwind
                elif archivo.startswith("tailwind.config."):
                    try:
                        with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                            limpio = limpiar_codigo(f.read())
                        texto += f"\n\n/* [{nombre}] {archivo} */\n{limpio}"
                    except Exception:
                        pass

        return texto

    except Exception as e:
        print(f"  ⚠ Error clonando {nombre}: {e}")
        return ""
    finally:
        if os.path.exists(ruta):
            shutil.rmtree(ruta, ignore_errors=True)


# ========================== CHECKPOINT =============================
def cargar_checkpoint() -> tuple[int, int]:
    """Devuelve (bytes_ya_escritos, repos_procesados) para reanudar."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                partes = f.read().strip().split(",")
                return int(partes[0]), int(partes[1])
        except Exception:
            pass
    return 0, 0


def guardar_checkpoint(bytes_escritos: int, repos_procesados: int):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(f"{bytes_escritos},{repos_procesados}")


# ========================== MAIN ===================================
if __name__ == "__main__":
    t_inicio = time.time()
    os.makedirs(TEMPORAL_DIR, exist_ok=True)

    # Reanudar si hay checkpoint
    bytes_previos, repos_previos = cargar_checkpoint()
    if repos_previos > 0:
        print(f"♻️  Reanudando desde repo #{repos_previos + 1} "
              f"({bytes_previos / 1e6:.1f} MB ya escritos)\n")

    repos = obtener_repositorios(CANTIDAD_REPOS, ESTRELLAS_MIN)
    repos_pendientes = repos[repos_previos:]

    modo = "ab" if repos_previos > 0 else "wb"
    archivos_ok, archivos_err = 0, 0

    with open(ARCHIVO_SALIDA, modo) as f_out:
        for i, url in enumerate(repos_pendientes, start=repos_previos + 1):
            texto = procesar_repositorio(url, i, len(repos), t_inicio)
            if texto:
                f_out.write(texto.encode("utf-8"))
                f_out.flush()
                archivos_ok += 1
            else:
                archivos_err += 1

            guardar_checkpoint(
                os.path.getsize(ARCHIVO_SALIDA),
                i,
            )

    # Limpiar checkpoint al terminar correctamente
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    if os.path.exists(TEMPORAL_DIR):
        shutil.rmtree(TEMPORAL_DIR, ignore_errors=True)

    mb     = os.path.getsize(ARCHIVO_SALIDA) / (1024 * 1024)
    minutos = (time.time() - t_inicio) / 60

    print("\n" + "=" * 60)
    print(f"✅ Dataset generado: {ARCHIVO_SALIDA}")
    print(f"   Repos OK     : {archivos_ok}")
    print(f"   Repos errores: {archivos_err}")
    print(f"   Tamaño       : {mb:.1f} MB")
    print(f"   Tiempo       : {minutos:.1f} min")
    print("=" * 60)
