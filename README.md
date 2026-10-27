# Ruido transcripcional como predictor de la patología de Alzheimer

**¿La variabilidad de expresión entre células predice la patología de Alzheimer mejor que la expresión media?**

La mayoría de los análisis de single-cell comparan la **expresión media** entre enfermos y controles. Pero una célula puede mantener su media y perder el control fino de su expresión: más **ruido transcripcional** de una célula a otra. Este proyecto mide ese ruido en núcleos de cerebro humano (snRNA-seq) y lo usa como variable de un modelo de aprendizaje automático para predecir la patología de Alzheimer de cada donante.

## Por qué importa (el "so what?")

Si el ruido aporta información **que la media no tiene**:

- señala **qué tipos celulares pierden antes la regulación** de su expresión, es decir, qué células son vulnerables primero;
- propone una **familia de biomarcadores** distinta de la expresión diferencial clásica;
- y ofrece una pregunta que se puede comprobar en datos independientes.

Si no aporta nada, también es un resultado útil: indica que basta con la media, que es más barata de medir.

## Cómo sé que el resultado es correcto

Esta es la parte central del proyecto. Con pocas decenas de donantes es muy fácil engañarse.

| Riesgo | Control en este pipeline |
|---|---|
| Pseudorreplicación: tratar células como muestras independientes | El modelo predice **donantes**, no células |
| La profundidad de secuenciación cambia el ruido aparente | Submuestreo de cada célula a la misma cantidad de UMIs |
| El número de células cambia la varianza estimada | Mismo número de células por donante y tipo celular |
| El CV² depende de la media de forma esperada | Se usa el **residuo** respecto a la tendencia media–CV² |
| Fuga de información al ajustar hiperparámetros | Validación cruzada **anidada**; escalado e imputación dentro del pipeline |
| Un AUC alto por azar | Test de permutación de etiquetas |
| El modelo solo aprende edad o sexo | Línea base con covariables |
| El código "funciona" pero no detecta nada | **Control positivo**: datos simulados con un efecto conocido solo en la varianza |

## Resultado con datos simulados (control positivo)

En la simulación, los donantes "enfermos" tienen más ruido en 40 genes de microglía **con la misma media**. El pipeline lo detecta con las features de ruido y no con las de media, que es lo esperado:

![Comparación de AUC](results/demo/auc_comparison.png)

Que la media salga algo por debajo de 0,5 es normal cuando no hay señal y hay pocos donantes: es ruido de la validación cruzada, no un efecto real.

## Datos reales

Pensado para el **Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD)**: snRNA-seq de la circunvolución temporal media con neuropatología por donante, descargable en formato `.h5ad` desde CELLxGENE. La etiqueta por defecto es el cambio neuropatológico de Alzheimer (ADNC: alto o intermedio frente a bajo o ninguno).

Antes de ejecutar, revisa los nombres de columna con `adata.obs.columns` y ajusta `config.yaml`.

## Uso

```bash
conda env create -f environment.yml
conda activate noise-ml-ad

python scripts/run_pipeline.py --demo        # prueba con datos simulados (~1-2 min)
python scripts/run_pipeline.py               # datos reales según config.yaml
python tests/test_noise.py                   # tests
```

Resultados en `results/`: AUC por partición (`cv_results.csv`), resumen (`report.json`), features con más peso y figuras.

## Estructura

```
src/noisead/
  data.py       carga de .h5ad y simulación con efecto conocido
  noise.py      submuestreo, estadísticos por grupo, residuo de CV², distancia al centroide
  features.py   tablas de features por donante (media, ruido, ruido global, combinadas)
  model.py      CV anidada repetida, test de permutación, coeficientes
  plots.py      figuras
scripts/run_pipeline.py
config.yaml
tests/
docs/PLAN.md    plan y calendario del proyecto
```

## Autora

Alba Pintado Rodrigo · Máster en Bioinformática y Ciencias de Datos en Medicina de Precisión (ISCIII). El proyecto parte de mi TFM en el CBMSO sobre ruido transcripcional en Alzheimer.
