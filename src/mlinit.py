import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =========================
# 1. Función para extraer características
# =========================

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estadísticos globales
    mean = np.mean(gray)
    std = np.std(gray)
    skewness = np.mean((gray - mean)*3) / (std*3 + 1e-8)
    kurtosis = np.mean((gray - mean)*4) / (std*4 + 1e-8)

    # Textura GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Frecuencias (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1e-8)
    freq_mean = np.mean(magnitude_spectrum)
    freq_std = np.std(magnitude_spectrum)

    return [mean, std, skewness, kurtosis,
            contrast, homogeneity, energy, correlation,
            freq_mean, freq_std]

# =========================
# 2. Construcción del dataset (multiclase)
# =========================

def build_multiclass_dataset(base_path, class_names):
    X, y = [], []
    for idx, cname in enumerate(class_names):
        folder = os.path.join(base_path, cname)
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname))
            if img is not None:
                X.append(extract_features(img))
                y.append(idx)
    return np.array(X), np.array(y)

# =========================
# 3. Entrenamiento y ranking
# =========================

class_names = ["sin_marca", "esquema1", "esquema2", "esquema3", "esquema4",
               "esquema5", "esquema6", "esquema7", "esquema8", "esquema9",
               "esquema10", "mio"]

X, y = build_multiclass_dataset("dataset", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Modelo
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Reporte clasificación en diccionario
report = classification_report(
    y_test, y_pred, target_names=class_names, output_dict=True
)

# Convertir a DataFrame
df_report = pd.DataFrame(report).transpose()
df_report = df_report.drop(["accuracy", "macro avg", "weighted avg"])

# Ordenar esquemas por "detectabilidad" (Recall alto = fácil de detectar)
df_report_sorted = df_report.sort_values("recall", ascending=True)

print("\n=== Ranking de esquemas (menos detectable → más detectable) ===")
print(df_report_sorted[["precision", "recall", "f1-score"]])

# =========================
# 4. Matriz de confusión
# =========================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Matriz de confusión - Random Forest")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()