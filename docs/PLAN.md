# Plan del proyecto: ruido transcripcional y ML en Alzheimer

## Pregunta

¿Las medidas de ruido transcripcional por tipo celular predicen la patología de Alzheimer de un donante, y aportan información **además** de la expresión media?

**Hipótesis:** algunas poblaciones (p. ej. microglía y neuronas vulnerables) aumentan su variabilidad célula a célula con la patología, y ese aumento se puede detectar aunque la media no cambie.

## So what

- Identifica qué tipos celulares pierden antes el control de su expresión.
- Propone el ruido como una clase de biomarcador complementaria a la expresión diferencial.
- Es comprobable: se puede validar en un segundo dataset independiente.

## Datos

| | Uso |
|---|---|
| **SEA-AD (MTG, snRNA-seq)** | Dataset principal. Neuropatología por donante (ADNC, Braak, Thal) y edad/sexo. Acceso abierto vía CELLxGENE. |
| **Segundo atlas público** (p. ej. Mathys et al., ROSMAP) | Validación externa, si el acceso lo permite. Revisar las condiciones de uso antes de descargar. |
| **Datos simulados** | Control positivo del pipeline (ya incluido). |

Si el atlas es muy grande, trabajar solo con los tipos celulares de interés y en modo `backed` de anndata.

## Métodos

1. **Preprocesado:** cuentas crudas, submuestreo a profundidad común, mismo número de células por donante x tipo celular, mínimo de células por grupo.
2. **Métricas de ruido:** residuo de log CV² respecto a la tendencia media–CV² (por gen) y distancia media al centroide (global). Comparar con las métricas del TFM y elegir las más robustas.
3. **Features por donante:** media (línea base), ruido por gen, ruido global, combinadas, covariables.
4. **Modelos:** regresión logística con regularización L1/L2 (principal, interpretable) y random forest (comparación).
5. **Evaluación:** CV anidada repetida, AUC; comparación por pares en las mismas particiones; test de permutación (≥200 permutaciones en el resultado final).
6. **Interpretación:** qué tipos celulares y genes pesan más; enriquecimiento funcional de esos genes.
7. **Extensiones si hay tiempo:** objetivo continuo (Braak o puntuación de progresión), validación externa, sensibilidad a la profundidad y al número de células.

## Calendario

| Fechas | Hito |
|---|---|
| Hasta 9 nov | Repo público con el pipeline y el control positivo simulado (v0.1). Enviar a Fernando Delgado-Chaves. |
| 10–23 nov | Descargar SEA-AD, explorar metadatos, primer cálculo de métricas en 2–3 tipos celulares. |
| 24 nov – 14 dic | Modelos con datos reales, controles técnicos, test de permutación. |
| 15 dic – 11 ene | Interpretación biológica, enriquecimiento, figuras finales. |
| 12–31 ene | README final con el "so what", limpieza de código, validación externa si es posible. |
| Febrero | Defensa del TFM. Repo terminado para recruiters y entrevistas. |

## Riesgos

- **Pocos donantes:** limita la potencia. Mitigación: modelos simples, CV repetida, intervalos.
- **Confusión técnica:** lote, intervalo post mortem, calidad del núcleo. Mitigación: incluirlos como covariables o comprobar que no predicen la etiqueta por sí solos.
- **Que el ruido no aporte nada:** también es un resultado publicable en el repo si está bien controlado.
- **Solapamiento con el TFM:** confirmar con María Llorens qué partes del código y de los resultados pueden ser públicas antes de subir nada con datos reales.

## Qué contar en una entrevista

1. La pregunta y por qué importa (30 s).
2. Cómo sé que el resultado es correcto: la tabla de controles del README.
3. El resultado y su límite principal.
4. Qué haría con más datos.
