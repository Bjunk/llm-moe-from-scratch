# Construí mi propio LLM en un Mac Mini. Sin millones de dólares. Y funciona.

Hay una narrativa instalada en el mundo de la IA que me molesta: que construir un modelo de lenguaje es territorio exclusivo de OpenAI, Google o Anthropic. Que necesitas clústeres de miles de GPUs, presupuestos de nueve cifras y equipos de investigadores con doctorado. Que el resto del mundo solo puede *usar* lo que ellos construyen.

Esa narrativa es parcialmente falsa. Y decidí demostrarlo.

---

## El punto de partida: ¿qué es realmente GPT-2?

En 2019, OpenAI publicó GPT-2. En ese momento fue un shock: un modelo que generaba texto coherente, completaba código, traducía idiomas. La versión más pequeña tenía 117 millones de parámetros. La más grande, 1.5 mil millones.

Hoy, esos números parecen modestos. GPT-4 tiene estimados de 1.8 *billones* de parámetros. Llama 3 tiene versiones de 70 mil millones. Los modelos comerciales modernos operan a una escala que efectivamente requiere infraestructura de cientos de millones de dólares.

Pero GPT-2 original — ese que en 2019 sorprendió al mundo — es perfectamente replicable hoy en hardware de consumo. Y yo quería ir un paso más allá: no solo replicar la *escala* de GPT-2, sino usar una arquitectura más moderna. Una que los grandes laboratorios usan en sus modelos más avanzados: **Mixture of Experts (MoE)**.

---

## El plan: enseñarle React a un modelo desde cero

Elegí un dominio deliberadamente acotado: **código React/TypeScript**. No intenté construir un modelo de propósito general — eso requiere billones de tokens de texto diverso. En cambio, quería demostrar que un modelo especializado, entrenado en un laptop, puede aprender patrones reales y no triviales de programación.

El pipeline completo tiene tres etapas:

**1. Datos: scraping de GitHub a escala**

Construí un scraper que extrae repositorios públicos de GitHub filtrados por lenguaje (JSX/TSX), estrellas mínimas (proxy de calidad), excluyendo archivos minificados, bundles y código auto-generado. El resultado: **137 MB de código React real**, proveniente de repositorios de producción. No tutoriales. Código que gente real usa en productos reales.

Después de tokenización: **24.4 millones de tokens**.

```python
# El scraper filtra por calidad antes de guardar
if repo["stargazers_count"] < 10:
    continue
if filename.endswith(".min.js") or "bundle" in filename:
    continue
```

**2. Tokenizador byte-level BPE**

Un detalle que la mayoría ignora: el tokenizador importa tanto como el modelo. El tokenizador clásico que corta por espacios rompe camelCase — `setCount` se tokeniza como `set` + `Count` y al decodificar aparece como `set Count`. En código eso es ruido devastador.

La solución: un tokenizador **byte-level BPE**, el mismo approach que usa GPT-2. Cada byte tiene representación propia. El tokenizador aprende qué secuencias de bytes son comunes en el corpus y las agrupa. El resultado:

```
setCount      → decode: 'setCount'    ✅
handleClick   → decode: 'handleClick' ✅
setIsLoading  → decode: 'setIsLoading' ✅
```

Vocabulario: 32.000 tokens. Tiempo de entrenamiento del tokenizador: **6 segundos**.

**3. La arquitectura: Mixture of Experts**

Aquí está la parte interesante. En vez de construir un Transformer clásico con capas FFN densas, implementé un **MoE real**.

La idea de MoE es elegante: en vez de que *todos* los parámetros de una capa procesen *cada* token, un router aprende a enviar cada token a los **k expertos más relevantes** (en mi caso, k=2 de 4 disponibles). Los demás expertos no se activan para ese token.

Esto significa que el modelo tiene más parámetros en total, pero *usa* menos por token. Es sparsity controlada.

```python
class CapaMoE(nn.Module):
    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)
        top_val, top_idx = torch.topk(probs, k=2, dim=-1)  # top-2 expertos
        gate = torch.zeros_like(probs).scatter_(1, top_idx, top_val)
        
        salida = sum(
            self.expertos[e](x) * gate[:, e:e+1]
            for e in range(self.num_expertos)
        )
        return salida
```

El router además tiene un **auxiliary loss** que penaliza cuando todos los tokens se van al mismo experto — sin eso, el modelo colapsa en un solo experto y el MoE no sirve de nada.

**Parámetros totales: 35.4 millones. Parámetros activos por token: 29.1 millones.**

Para contexto: GPT-2 small tiene 117M. Mi modelo tiene 3× menos. Mixtral 8×7B tiene 47 mil millones totales. Estoy en un orden de magnitud completamente distinto — y eso es exactamente el punto.

---

## El hardware: un Mac Mini M4

Un Mac Mini M4 con 16 GB de memoria unificada. Sin GPU dedicada en el sentido clásico. Sin nube. Sin factura de AWS al final del mes.

El chip M4 de Apple tiene un acelerador de ML integrado accesible vía **Metal Performance Shaders (MPS)**. PyTorch lo soporta de forma nativa. El entrenamiento corre directamente en el chip.

Algunas decisiones de ingeniería que hicieron posible entrenar en MPS sin quedarse sin memoria:

- **Batch físico de 32** con gradient accumulation de 8 pasos (batch efectivo de 256). Sin esto, el tensor de logits `(256 × 256 × 32.000)` agota los 20 GB disponibles.
- **MoE con matmul denso** en vez de indexado sparse. En CUDA el indexado sparse es eficiente. En Metal, paralizar matmuls densos es más rápido que el overhead del indexado booleano.
- **Weight tying**: los pesos del embedding y la capa de salida son el mismo tensor. Ahorra ~16M de parámetros sin costo de calidad.

---

## Los resultados: el modelo aprende

Epoch 1. 11.331 lotes. Aproximadamente 2.5 horas en el M4.

La perplejidad de validación — la métrica que indica qué tan bien generaliza el modelo a código que *no vio durante el entrenamiento* — cayó de **14.731 a 36.1**.

```
[Época 1] Lote  500  → CE: 9.60 | ppl: 14,731
[Época 1] Lote 3000  → CE: 5.64 | ppl:    281
[Época 1] Lote 7000  → CE: 3.99 | ppl:     54
[Época 1] Lote 11000 → CE: 2.78 | ppl:     16
📊 Época 1 → val CE: 3.59 | val ppl: 36.1 ✅
```

Val ppl 36.1 para un modelo de 35M params entrenado desde cero. Para referencia: GPT-2 de 117M (3× más grande) logró ~29 en texto general. Estamos en el mismo orden de magnitud, en un dominio especializado, con un tercio de los parámetros.

Pero la perplejidad es un número abstracto. Lo que realmente importa: **¿qué genera el modelo?**

Le di el prompt `export default function App() {` y esto es lo que produjo:

```tsx
export default function App() {
  return (
    <Refine KBar>
      <Product />
    </Refine>
    <ChakraProvider>
  );
}

import React from "react";
import type { RefineThemedLayoutProvider } from "@refinedev/core";
import dataProvider from "@refinedev/simple-rest";

const API_URL = "https://...
```

El modelo no solo sabe que `App()` devuelve JSX. Aprendió que existe un framework llamado **Refine**, que se usa con **ChakraProvider**, que los imports van arriba, que `dataProvider` viene de `@refinedev/simple-rest`. Eso no está en ninguna regla explícita. Emergió del corpus.

Con `useEffect(() => {` generó:

```tsx
useEffect(() => {
  if (command && !posthog.capture('signup', {
    id: "auth",
    description: 'Auth email with password',
  })) return;
  
  const api = await Promise.all([
    generateApp({ project }),
  ]);
}, [router, params]);
```

Conoce **posthog** (plataforma de analytics). Conoce `Promise.all`. Conoce el patrón de dependency array en `useEffect`. Conoce async/await dentro de effects. Todo aprendido de repositorios reales.

---

## Lo que esto demuestra — y lo que no

Seré directo: esto no compite con GPT-4, Claude o Gemini. No va a razonar, no va a explicar conceptos, no va a depurar tu código con coherencia de múltiples pasos. Para eso se necesita escala que está genuinamente fuera del alcance de un Mac Mini.

Lo que sí demuestra es más importante:

**1. La democratización de la IA no es solo API.** Entender cómo funcionan estos sistemas — el routing de MoE, el auxiliary loss, el weight tying, por qué el tokenizador importa — es conocimiento que antes requería años de academia. Hoy se puede experimentar en un fin de semana.

**2. Los modelos especializados tienen un futuro real.** Un modelo de 35M entrenado específicamente en código React de producción ya muestra patrones que un modelo generalista de la misma escala no tendría. La especialización compensa la escala hasta cierto punto.

**3. La arquitectura MoE es accesible.** Mixtral, DeepSeek, Grok — todos usan MoE. No es magia de laboratorio. Son decisiones de ingeniería replicables con PyTorch y paciencia.

---

## El siguiente paso

Con 35M params y 24M tokens, llegamos al límite de lo que tiene sentido entrenar desde cero en hardware de consumo. El siguiente movimiento natural no es escalar el modelo propio — es usar este conocimiento para hacer **fine-tuning** de un modelo base que ya tiene inteligencia incorporada.

Qwen2.5-Coder-7B o DeepSeek-Coder entrenaron sobre cientos de miles de millones de tokens. Con LoRA y un g5.xlarge en AWS a $1.21/hr, se puede especializar ese modelo en cualquier dominio con datos propios. El costo de un run completo: menos de $50.

Eso es lo que OpenAI gastó en café en 2019. Y hoy está al alcance de cualquier founder técnico dispuesto a entender qué está haciendo.

---

*¿Quieres el código completo? Está todo en Python, sin dependencias exóticas, con comentarios en cada decisión de arquitectura. La barrera de entrada real no es el dinero — es el tiempo de entender qué está pasando dentro de la caja negra.*

*De Cero a Unicornio — Pablo*
