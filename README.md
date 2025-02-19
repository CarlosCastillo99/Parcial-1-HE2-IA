# Proyecto de Predicción de Diabetes en Población Indígena Pima

Este repositorio contiene un proyecto de análisis y predicción de diabetes en la población indígena Pima utilizando técnicas de aprendizaje automático. El objetivo principal es identificar los factores más relevantes asociados con la presencia de diabetes y construir modelos predictivos para facilitar el diagnóstico temprano de la enfermedad.

## Objetivos del Proyecto

El proyecto tiene los siguientes objetivos:

1. **Análisis Exploratorio de Datos (EDA):** Realizar un análisis descriptivo de las variables médicas y demográficas para comprender su distribución, correlaciones y posibles relaciones con la variable objetivo (*Outcome*).
2. **Preprocesamiento de Datos:** Manejar valores faltantes, escalar características y balancear el conjunto de datos utilizando técnicas como SMOTE.
3. **Modelado Predictivo:** Entrenar y evaluar varios modelos de clasificación para predecir la presencia de diabetes en pacientes.
4. **Evaluación de Modelos:** Comparar el rendimiento de los modelos utilizando métricas como precisión, *recall*, F1-Score y AUC-ROC.

## Dataset

El conjunto de datos utilizado es el **Pima Indians Diabetes Database**, que contiene información médica de mujeres indígenas Pima mayores de 21 años. Las variables incluyen:

- **Pregnancies:** Número de embarazos.
- **Glucose:** Nivel de glucosa en sangre (mg/dL).
- **BloodPressure:** Presión arterial diastólica (mm Hg).
- **SkinThickness:** Espesor del pliegue cutáneo del tríceps (mm).
- **Insulin:** Nivel de insulina en suero (mu U/ml).
- **BMI:** Índice de Masa Corporal (peso en kg / altura en m²).
- **DiabetesPedigreeFunction:** Función de pedigrí de diabetes (probabilidad de desarrollar diabetes).
- **Age:** Edad en años.
- **Outcome:** Diagnóstico de diabetes (0 = No, 1 = Sí).

## Modelos Utilizados

Se entrenaron y evaluaron los siguientes modelos de clasificación:

- Regresión Logística
- SVM (Máquinas de Vectores de Soporte)
- Random Forest
- XGBoost
- KNN (Vecinos más Cercanos)

## Requisitos

Para ejecutar este proyecto en tu máquina local, asegúrate de tener las siguientes librerías instaladas:

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn ydata_profiling
```

## Cómo Ejecutar el Proyecto

1. **Clonar el Repositorio:**

   ```bash
   git clone https://github.com/tu-usuario/nombre-del-repositorio.git
   cd nombre-del-repositorio
   ```

2. **Ejecutar el Cuaderno de Colab:**

   Abre el archivo `Copia_de_Monitoria_Parcial1.ipynb` en Google Colab o en tu entorno local con Jupyter Notebook.

3. **Preprocesamiento y Modelado:**

   Sigue los pasos detallados en el cuaderno para realizar el análisis exploratorio, preprocesamiento de datos, entrenamiento de modelos y evaluación.

## Ejemplos de Uso

### Análisis Exploratorio

El análisis exploratorio incluye:

- Visualización de la distribución de variables.
- Detección de valores atípicos.
- Matriz de correlación para identificar relaciones entre variables.

### Preprocesamiento

- **Imputación de Valores Faltantes:** Se reemplazan los valores faltantes en variables como `BloodPressure`, `SkinThickness`, `Insulin` y `BMI` con la media de la columna.
- **Escalado de Características:** Se aplica `StandardScaler` para normalizar las variables.
- **Balanceo de Clases:** Se utiliza SMOTE para generar ejemplos sintéticos de la clase minoritaria (diabetes).

### Modelado y Evaluación

- **Entrenamiento de Modelos:** Se entrenan varios modelos de clasificación utilizando `GridSearchCV` para optimizar hiperparámetros.
- **Evaluación:** Se comparan los modelos utilizando métricas como precisión, *recall*, F1-Score y AUC-ROC.

## Resultados

Los resultados del proyecto incluyen:

- **Comparación de Métricas:** Se presenta una comparación de las métricas de rendimiento para cada modelo.
- **Curva ROC:** Se grafica la curva ROC para evaluar la capacidad de discriminación de los modelos.
- **Log Loss:** Se calcula la entropía cruzada para evaluar la calidad de las predicciones probabilísticas.

## Contribuciones

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una nueva rama (`git checkout -b feature-nueva`).
3. Realiza tus cambios y haz commit de ellos (`git commit -am 'Agrega nueva funcionalidad'`).
4. Envía tus cambios al repositorio remoto (`git push origin feature-nueva`).
5. Abre un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo `LICENSE`.

---

Este proyecto fue desarrollado como parte de un trabajo de Inteligencia Artificial aplicada a la economía. Para cualquier pregunta o sugerencia, no dudes en contactar al autor.

