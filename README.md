# **Clasificación de Diabetes Tipo 2 - Dataset Pima Indians**

## **Descripción del Proyecto**
Este proyecto utiliza **Machine Learning** para predecir la presencia de **diabetes tipo 2** en mujeres de la comunidad **Pima Indians** utilizando un conjunto de datos de **Kaggle**. Se implementaron modelos de clasificación avanzados junto con técnicas de balanceo de clases como **SMOTE** para mejorar la precisión en la detección de casos positivos (diabetes).

El enfoque principal es optimizar las métricas de **Recall Weighted** y **AUC-ROC** para minimizar los **falsos negativos**, ya que en el contexto médico es crucial detectar correctamente los casos positivos.

---

## **Objetivo del Proyecto**
El objetivo principal es desarrollar un modelo de clasificación robusto para predecir diabetes tipo 2, utilizando un enfoque **end-to-end** que incluye:
- Análisis Exploratorio de Datos (EDA) para entender las características y distribuciones de las variables.
- Preprocesamiento de datos, incluyendo manejo de valores atípicos y datos faltantes.
- Balanceo de clases con **SMOTE** para mejorar el rendimiento en la clase minoritaria.
- Entrenamiento y evaluación de varios modelos de clasificación.
- Análisis de métricas avanzadas como **Log Loss** y **Entropía** para evaluar la certeza de las predicciones.

---

## **Dataset**
- **Fuente:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Descripción:**
  - 768 observaciones con 9 variables predictoras y una variable objetivo (`Outcome`).
  - Variables predictoras incluyen:
    - `Pregnancies`: Número de embarazos.
    - `Glucose`: Concentración de glucosa en sangre.
    - `BloodPressure`: Presión arterial diastólica.
    - `SkinThickness`: Grosor del pliegue cutáneo.
    - `Insulin`: Niveles de insulina en suero.
    - `BMI`: Índice de Masa Corporal.
    - `DiabetesPedigreeFunction`: Antecedentes familiares de diabetes.
    - `Age`: Edad.
  - `Outcome`: Variable objetivo binaria (0 = No Diabetes, 1 = Diabetes).

---

## **Modelos de Clasificación Utilizados**
Se evaluaron y compararon los siguientes algoritmos de clasificación:
- **Regresión Logística (Ridge y Lasso)**
- **SVM (Support Vector Machine)**
- **Árbol de Decisión (Decision Tree)**
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**

---

## **Metodología**
### **1. Análisis Exploratorio de Datos (EDA)**
- Análisis de distribuciones y correlaciones.
- Identificación de valores atípicos y datos faltantes.
- Visualización de relaciones entre variables predictoras y la variable objetivo.

### **2. Preprocesamiento de Datos**
- **Imputación de datos faltantes** en `BloodPressure`, `SkinThickness` e `Insulin` utilizando la **media**.
- **Estandarización** de variables con `StandardScaler()` para mejorar el rendimiento de modelos basados en distancia.

### **3. Balanceo de Clases con SMOTE**
- Aplicación de **SMOTE** para balancear las clases en el conjunto de entrenamiento.
- Comparación de métricas antes y después de aplicar SMOTE para evaluar el impacto en el rendimiento de los modelos.

### **4. Evaluación de Modelos**
- **Métricas Tradicionales:**
  - `Accuracy`, `Precision`, `Recall`, `F1-Score`.
- **Métricas Avanzadas:**
  - `AUC-ROC`: Para evaluar la capacidad discriminativa de los modelos.
  - `Log Loss`: Para medir la incertidumbre en las probabilidades predichas.
  - `Entropía`: Para analizar la certeza de las predicciones.

### **5. Visualización de Resultados**
- **Curvas ROC** para comparar la capacidad de discriminación de los modelos.
- **Gráficos de barras** para comparar métricas tradicionales.
- **Distribución de Entropía** para evaluar la incertidumbre de las predicciones.

---

## **Resultados y Análisis**
- **Mejor modelo antes de SMOTE:** `KNN` con **Recall Weighted = 0.753**.
- **Mejor modelo después de SMOTE:** `Random Forest` con **AUC-ROC = 0.823** y **Log Loss = 0.483**.
- **Impacto de SMOTE:**
  - Mejoró el rendimiento en Recall para la clase minoritaria (Diabetes).
  - Disminuyó la entropía y la incertidumbre en las predicciones de casi todos los modelos.
- **Entropía:**
  - `Decision Tree` mostró la menor entropía pero puede estar **sobreajustado**.
  - `XGBoost` y `KNN` mantuvieron un buen balance entre certeza e incertidumbre.

---

## **Conclusiones y Recomendaciones**
- **Random Forest** es el modelo más robusto para predecir diabetes tipo 2 en este dataset, mostrando una excelente capacidad discriminativa.
- **SMOTE** mejoró significativamente el rendimiento en la clase minoritaria, destacando la importancia del balance de clases en problemas de clasificación médica.
- Se recomienda ajustar umbrales de decisión en **XGBoost** y **KNN** para optimizar la precisión en escenarios con mayor incertidumbre.
- Considerar el uso de **CalibratedClassifierCV** para mejorar las probabilidades en **SVM** y reducir la entropía indefinida.

---

## **Requisitos de Instalación**
Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes bibliotecas:
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```
---
## Autores
- **Carlos Castillo** - Autor principal
- **Jennifer Sanabria** - Autor principal
- **Juan Saldaña** - Autor principal
- **Ángela Torres** - Autor principal

---

## Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo `LICENSE`.

---

## Referencias
- **Dataset**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **SMOTE**: Chawla et al. (2002). *"SMOTE: Synthetic Minority Over-sampling Technique"*.
- **XGBoost**: Chen & Guestrin (2016). *"XGBoost: A Scalable Tree Boosting System"*.



