import os
import re
import shutil
import requests
import time
from git import Repo

# ==========================================
# CONFIGURACIÓN DEL EXTRACTOR HÍBRIDO V4
# ==========================================
GITHUB_TOKEN = "your-github-pat" 

CANTIDAD_REPOS = 100        
ESTRELLAS_MINIMAS = 500    
TEMPORAL_DIR = "./temp_repos"
ARCHIVO_SALIDA = "mega_dataset_react.txt"

# 📂 TU RUTA LOCAL (Apuntando directo al desarrollo de Califix)
RUTA_LOCAL_CALIFIX = "/Users/pablo/your-local-path/code"

ruido_carpetas = {"node_modules", ".next", "dist", "cypress", ".git", "build", "public", "out", "coverage", "scripts"}
ruido_archivos = {"test", "spec", "webpack", "eslint", "jest", "changelog", "license"}

def obtener_repositorios_top():
    print(f"🔍 Buscando los {CANTIDAD_REPOS} repositorios más populares de React/Next.js en GitHub...")
    url = f"https://api.github.com/search/repositories?q=topic:nextjs+topic:react+stars:>{ESTRELLAS_MINIMAS}&sort=stars&order=desc&per_page={CANTIDAD_REPOS}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            items = response.json().get("items", [])
            return [item["clone_url"] for item in items]
        print(f"❌ Error API GitHub: {response.status_code}")
        return []
    except Exception as e:
        print(f"❌ Error de red: {e}")
        return []

def limpiar_codigo(codigo):
    codigo = re.sub(r"//.*", "", codigo)
    codigo = re.sub(r"/\*.*?\*/", "", codigo, flags=re.DOTALL)
    codigo = re.sub(r"[ \t]+", " ", codigo)
    codigo = re.sub(r"\n\s*\n", "\n", codigo)
    return codigo.strip()

# 🚀 NUEVA FUNCIÓN: MINERÍA QUIRÚRGICA DE TU PROPIO CÓDIGO LOCAL
def extraer_codigo_local(ruta_raiz):
    print(f"\n🏠 [LOCAL] Iniciando escaneo de componentes nativos en: {ruta_raiz}...")
    texto_local = ""
    conteo_archivos = 0
    conteo_tailwind = 0
    
    if not os.path.exists(ruta_raiz):
        print(f"⚠️ Alerta: La ruta local '{ruta_raiz}' no existe. Saltando inyección local.")
        return ""

    for raiz, dirs, archivos in os.walk(ruta_raiz):
        # Evitar carpetas de compilación o dependencias locales
        dirs[:] = [d for d in dirs if d.lower() not in ruido_carpetas and not d.startswith(".")]
        
        for archivo in archivos:
            ruta_completa = os.path.join(raiz, archivo)
            
            # Capturar estrictamente tus interfaces JSX/TSX
            if archivo.endswith((".jsx", ".tsx")):
                if not any(ruido in archivo.lower() for ruido in ruido_archivos):
                    try:
                        with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                            contenido_limpio = limpiar_codigo(f.read())
                            if len(contenido_limpio) > 30:
                                # Le metemos una etiqueta especial para que el modelo entienda que es código Califix
                                texto_local += f"\n\n/* Califix Native Component: {archivo} */\n" + contenido_limpio
                                conteo_archivos += 1
                    except Exception:
                        pass
            
            # Capturar tus archivos de configuración de diseño locales
            elif archivo.startswith("tailwind.config."):
                try:
                    with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                        contenido_limpio = limpiar_codigo(f.read())
                        texto_local += f"\n\n/* Califix Local Tailwind Config: {archivo} */\n" + contenido_limpio
                        conteo_tailwind += 1
                except Exception:
                    pass

    print(f"   📥 Extracción local completada: +{conteo_archivos} componentes UI y +{conteo_tailwind} configs de Tailwind inyectados con éxito.")
    return texto_local

def procesar_repositorio(url_repo, index, total):
    nombre_repo = url_repo.split("/")[-1].replace(".git", "")
    ruta_clonado = os.path.join(TEMPORAL_DIR, nombre_repo)
    print(f"🌐 [{index}/{total}] Clonando: {nombre_repo}...")
    if os.path.exists(ruta_clonado): shutil.rmtree(ruta_clonado)
    try:
        Repo.clone_from(url_repo, ruta_clonado, depth=1, single_branch=True)
        texto_extraido = ""
        for raiz, dirs, archivos in os.walk(ruta_clonado):
            dirs[:] = [d for d in dirs if d.lower() not in ruido_carpetas and not d.startswith(".")]
            for archivo in archivos:
                ruta_completa = os.path.join(raiz, archivo)
                if archivo.endswith((".jsx", ".tsx")):
                    if not any(ruido in archivo.lower() for ruido in ruido_archivos):
                        try:
                            with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                                contenido_limpio = limpiar_codigo(f.read())
                                if len(contenido_limpio) > 50:
                                    texto_extraido += f"\n\n/* Global Component: {archivo} */\n" + contenido_limpio
                        except Exception:
                            pass
                elif archivo.startswith("tailwind.config."):
                    try:
                        with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                            texto_extraido += f"\n\n/* Global Tailwind Config: {archivo} */\n" + limpiar_codigo(f.read())
                    except Exception:
                        pass
        return texto_extraido
    except Exception:
        return ""
    finally:
        if os.path.exists(ruta_clonado): shutil.rmtree(ruta_clonado)

if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists(TEMPORAL_DIR): os.makedirs(TEMPORAL_DIR)
        
    codigo_total_acumulado = ""
    
    # 1. Ejecutar la inyección del ADN de tus plataformas locales
    codigo_total_acumulado += extraer_codigo_local(RUTA_LOCAL_CALIFIX)
    
    # 2. Descargar la cultura general de internet para complementar
    repos_api = obtener_repositorios_top()
    if repos_api:
        for i, repo_url in enumerate(repos_api):
            codigo_total_acumulado += procesar_repositorio(repo_url, i + 1, len(repos_api))
        
    if codigo_total_acumulado:
        with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f_out:
            f_out.write(codigo_total_acumulado)
            
        tamano_mb = os.path.getsize(ARCHIVO_SALIDA) / (1024 * 1024)
        print("\n" + "="*60)
        print(f"🎉 ¡SÚPER DATASET HÍBRIDO LOCAL-GLOBAL GENERADO!")
        print(f"📦 Archivo consolidado: '{ARCHIVO_SALIDA}'")
        print(f"💾 Tamaño refinado en disco: {tamano_mb:.2f} MB")
        print(f"⏱️ Tiempo total de ejecución: {(time.time() - start_time) / 60:.2f} minutos")
        print("="*60 + "\n")
        
    if os.path.exists(TEMPORAL_DIR): shutil.rmtree(TEMPORAL_DIR, ignore_errors=True)
