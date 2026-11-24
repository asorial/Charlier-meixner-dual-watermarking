import os
import numpy as np
import cv2
import torch
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage import color
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, 
                            classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from piq import brisque
from tqdm import tqdm

# Ignorar advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning, module="piq")

# =========================
# 1. Función mejorada para extraer características
# =========================

def extract_advanced_features(img):
    # Convertir a diferentes espacios de color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Inicializar lista de características
    features = []
    
    # 1. Estadísticas básicas (RGB + Gray)
    for channel, name in zip([img[:,:,0], img[:,:,1], img[:,:,2], gray], 
                            ['B', 'G', 'R', 'Gray']):
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(skew(channel.ravel()))
        features.append(kurtosis(channel.ravel()))
    
    # 2. Estadísticas en HSV y LAB
    for space, channels in zip([hsv, lab], ['HSV', 'LAB']):
        for i in range(3):
            ch = space[:,:,i]
            features.append(np.mean(ch))
            features.append(np.std(ch))
    
    # 3. Textura GLCM (4 direcciones)
    glcm = graycomatrix(gray, distances=[1, 3], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    
    texture_props = ['contrast', 'homogeneity', 'energy', 'correlation']
    for prop in texture_props:
        val = np.mean(graycoprops(glcm, prop))
        features.append(val)
    
    # 4. Dominio de frecuencia (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8
    features.append(np.mean(magnitude_spectrum))
    features.append(np.std(magnitude_spectrum))
    
    # 5. DCT (Transformada Coseno Discreta)
    dct = cv2.dct(np.float32(gray)/255.0)
    dct_energy = np.sum(dct**2)
    dct_high_freq = np.sum(dct[int(dct.shape[0]*0.7):, int(dct.shape[1]*0.7):]**2)
    features.extend([dct_energy, dct_high_freq])
    
    # 6. Métricas de calidad (BRISQUE)
    try:
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        brisque_score = brisque(img_tensor, data_range=1.0)
        features.append(brisque_score.item())
    except:
        features.append(0)  # Valor por defecto si falla
    
    # 7. Correlación entre canales
    corr_rg = np.corrcoef(img[:,:,2].ravel(), img[:,:,1].ravel())[0,1]
    corr_rb = np.corrcoef(img[:,:,2].ravel(), img[:,:,0].ravel())[0,1]
    features.extend([corr_rg, corr_rb])
    
    # 8. Residual Analysis
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    residual = gray - blurred
    features.append(np.std(residual))
    
    return np.array(features)

# =========================
# 2. Construcción del dataset con estructura mejorada
# =========================

def build_scheme_dataset(base_path, class_names):
    X, y, schemes = [], [], []
    
    # Procesar todas las clases
    for scheme in class_names:
        folder = os.path.join(base_path, scheme)
        print(f"Procesando esquema: {scheme}")
        
        for fname in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is not None:
                # Redimensionar para consistencia (opcional)
                img = cv2.resize(img, (256, 256))
                
                # Extraer características
                features = extract_advanced_features(img)
                
                # Etiqueta binaria: 0 = sin marca, 1 = con marca
                label = 0 if scheme == "sin_marca" else 1
                
                X.append(features)
                y.append(label)
                schemes.append(scheme)
    
    return np.array(X), np.array(y), np.array(schemes)

# =========================
# 3. Configuración principal
# =========================
class_names = ["sin_marca", "esquema1", "esquema2", "esquema3", "esquema4",
               "esquema5", "esquema6", "esquema7", "esquema8", "esquema9",
               "esquema10", "mio"]

# Construir dataset
X, y, schemes = build_scheme_dataset("dataset", class_names)

# Dividir datos conservando la distribución por esquema
X_train, X_test, y_train, y_test, schemes_train, schemes_test = train_test_split(
    X, y, schemes, test_size=0.3, random_state=42, stratify=schemes
)

# =========================
# 4. Entrenamiento del modelo
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# =========================
# 5. Análisis por esquema
# =========================
scheme_results = []
plt.figure(figsize=(12, 8))

# Almacenar resultados por esquema
for scheme in class_names:
    if scheme == "sin_marca":  # No evaluar el esquema base
        continue
        
    # Filtrar datos del esquema actual
    mask = (schemes_test == scheme) | (schemes_test == "sin_marca")
    X_scheme = X_test[mask]
    y_scheme = y_test[mask]
    
    # Predecir probabilidades
    y_prob = rf.predict_proba(X_scheme)[:, 1]
    
    # Calcular métricas
    auc = roc_auc_score(y_scheme, y_prob)
    fpr, tpr, _ = roc_curve(y_scheme, y_prob)
    
    # Almacenar resultados
    scheme_results.append({
        'scheme': scheme,
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr
    })
    
    # Plotear curva ROC individual
    plt.plot(fpr, tpr, lw=2, alpha=0.5, label=f'{scheme} (AUC = {auc:.3f})')

# Configurar gráfico ROC
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC Comparativas por Esquema')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 6. Ranking de esquemas por detectabilidad
# =========================
# Crear DataFrame con resultados
df_results = pd.DataFrame(scheme_results)
df_results = df_results.sort_values('auc', ascending=True)

# Gráfico de barras para AUC
plt.figure(figsize=(14, 8))
bars = plt.barh(df_results['scheme'], df_results['auc'], color='skyblue')
plt.xlabel('AUC (Area Under ROC Curve)')
plt.title('Detectabilidad de Esquemas de Marca de Agua')
plt.xlim([0.4, 1.0])
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadir valores de AUC a las barras
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('auc_comparison.png', dpi=300)
plt.show()

# =========================
# 7. Importancia de características
# =========================
feature_importances = rf.feature_importances_
sorted_idx = np.argsort(feature_importances)[-20:]  # Top 20 características

plt.figure(figsize=(12, 10))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
plt.xlabel('Importancia')
plt.title('Top 20 Características Más Importantes')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# =========================
# 8. Resultados finales en tabla
# =========================
print("\n=== Ranking de Esquemas (Menos Detectable → Más Detectable) ===")
print(df_results[['scheme', 'auc']].reset_index(drop=True))

# Exportar resultados a CSV
df_results.to_csv('watermark_scheme_comparison.csv', index=False)