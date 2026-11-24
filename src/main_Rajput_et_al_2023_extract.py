
from helpers import progress_bar
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from scipy.linalg import hessenberg, svd
from scipy.linalg import svd
from PIL import Image
from skimage.util import view_as_blocks
from helpers.utils import *
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from pywt import dwt2, idwt2
import pywt
import numpy as np
import cv2
import os


alpha_par = 0.002  # Valor por defecto para alpha

# === Inserción ===

def embed_watermark(cover_rgb: np.ndarray, watermark: np.ndarray, alpha: float = alpha_par):
    """
    Inserta una marca de agua en el canal rojo de una imagen RGB usando
    DWT, HD y SVD según el artículo adjunto.
    Retorna la imagen marcada, y las matrices Uw, Vw y HSw_modified necesarias
    para la extracción.
    """
    R, G, B = cv2.split(cover_rgb)

    # DWT sobre canal R (1 nivel)
    coeffs2 = pywt.dwt2(R.astype(np.float64), 'haar')
    LL, (LH, HL, HH) = coeffs2

    # HD sobre LL. Obtener la matriz H de Hessenberg y la matriz unitaria Q
    H, Q = hessenberg(LL, calc_q=True)

    # SVD sobre H y la marca de agua
    HUw, HSw, HVw = svd(H)
    Uw, Sw, Vw = svd(watermark.astype(np.float64))

    # Modificar los valores singulares de H
    # Se crea una copia y se modifican los primeros N valores (donde N es el tamaño de Sw)
    HSw_modified = HSw.copy()
    HSw_modified[:len(Sw)] += alpha * Sw

    # Reconstruir la subbanda H modificada
    H_modified = HUw @ np.diag(HSw_modified) @ HVw

    # Reconstruir la subbanda LL modificada usando la matriz Q
    # LL_modified = Q * H_modified * Q' (Q' es la transpuesta de Q)
    LL_modified = Q @ H_modified @ Q.T

    # Reconstruir canal R
    coeffs2_marked = (LL_modified, (LH, HL, HH))
    R_marked = pywt.idwt2(coeffs2_marked, 'haar')
    R_marked = np.clip(R_marked, 0, 255).astype(np.uint8)

    # Recomponer imagen RGB
    watermarked_rgb = cv2.merge((R_marked, G, B))
    return watermarked_rgb, Uw, Vw


# ===================================================

# === Extracción de marca de agua ===

def extract_watermark(watermarked_rgb: np.ndarray, original_rgb: np.ndarray, Uw: np.ndarray, Vw: np.ndarray, alpha: float = alpha_par) -> np.ndarray:
    """
    Extrae la marca de agua del canal rojo según el artículo adjunto.
    Utiliza las matrices Uw y Vw de la marca de agua original.
    """
    # Extraer el canal rojo de la imagen con marca de agua y de la original
    Rw = cv2.split(watermarked_rgb)[0].astype(np.float64)
    Ro = cv2.split(original_rgb)[0].astype(np.float64)

    # DWT (1 nivel) sobre los canales rojos
    LLw, _ = pywt.dwt2(Rw, 'haar')
    LLo, _ = pywt.dwt2(Ro, 'haar')

    # HD sobre los sub-canales LL
    Hw, Qw = hessenberg(LLw, calc_q=True)
    Ho, Qo = hessenberg(LLo, calc_q=True)

    # SVD sobre las matrices de Hessenberg
    HUw_ext, HSw_ext, HVw_ext = svd(Hw)
    HUw_orig, HSw_orig, HVw_orig = svd(Ho)

    # El tamaño de la marca de agua es la longitud de la matriz de valores singulares de la marca de agua
    watermark_size = Uw.shape[0]

    # Reconstruir los valores singulares de la marca de agua
    Sw_ext = (HSw_ext[:watermark_size] - HSw_orig[:watermark_size]) / alpha

    # Reconstruir la marca de agua usando las matrices U y V de la marca de agua original
    watermark_extracted = Uw @ np.diag(Sw_ext) @ Vw
    return np.clip(watermark_extracted, 0, 255).astype(np.uint8)


# ===== Main Function =====

def main():
    # Instancia para métricas
    metr = Metrics()

    # Directorios base
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_attacked = os.path.join(base, "watermarked_with_nosises")

    # Directorio de marcas de agua
    watermark_path = os.path.join(base, "watermark")
    watermark_path = os.path.join(
        watermark_path, os.listdir(watermark_path)[0]
    )

    # Leer marca de agua
    watermark_array = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark_bin = (np.array(watermark) > 127).astype(np.uint8)

    sub_ind = ['10']

    for value in sub_ind:

        # Nombre del método
        method_name = f"Rajput_{value}"
        print(f"Processing method: {method_name}")
        dir_method = os.path.join(dir_attacked, method_name)

        if not os.path.exists(dir_method):
            print(f"Directory does not exist: {dir_method}")
            continue

        # Listar ruidos/distorsiones aplicadas
        noises_list = os.listdir(dir_method)
        noises_list = sorted(noises_list, key=lambda x: int(x.split("_")[0]))

        # Barra de progreso
        progress_bar.printProgressBar(0, len(noises_list), prefix='Progress:', suffix='Complete', length=50)

        for idx, noise_name in enumerate(noises_list):
            dir_noise = os.path.join(dir_method, noise_name)
            levels = os.listdir(dir_noise)
            levels = sorted(levels, key=lambda x: int(x))

            for level in levels:
                dir_level = os.path.join(dir_noise, level)
                image_names = os.listdir(dir_level)

                # Ordenar imágenes numéricamente
                image_names = sorted(image_names, key=lambda x: int(x.split("_")[1].split('.')[0]))

                # Listas para almacenar métricas
                list_ber = []
                list_ncc = []

                for img_idx, image_name in enumerate(image_names):
                    # Cargar imagen atacada
                    img_path = os.path.join(dir_level, image_name)
                    attacked_img = cv2.imread(img_path)

                    # Cargar imagen original correspondiente
                    original_img_path = os.path.join(
                        base, "cover_images",
                        image_name.split('_')[0] + ".bmp"  # Asumiendo que las imágenes originales están en cover_images
                    )

                    # Verificar si la imagen original existe
                    if not os.path.exists(original_img_path):
                        print(f"Original image not found: {original_img_path}")
                        continue
                    original_img = cv2.imread(original_img_path)

                    _, Uw, Vw = embed_watermark(
                        original_img,
                        watermark_bin
                    )

                    # Extraer marca de agua
                    extracted_watermark = extract_watermark(
                        attacked_img,
                        original_img,
                        Uw, Vw
                    )

                    # Guardar marca extraída
                    save_dir = os.path.join(base, "watermark_ext_with_noise", method_name, noise_name, level)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, image_name)

                    # Save as image
                    cv2.imwrite(save_path, extracted_watermark)

                    # Calcular métricas
                    ber, ncc = metr.calculate_ber_ncc(watermark_array, extracted_watermark)

                    list_ber.append(ber)
                    list_ncc.append(ncc)

                    print(f"\nExperimental analysis ({method_name}, {noise_name}, {level}, {image_name})")
                    print(" ")
                    print(f"BER: {ber}")
                    print(" ")
                    print(f"NCC: {ncc}")
                    print(" ")

                    # Guardar resultados temporalmente
                    ber_dir = os.path.join(base, "results", "ber", method_name, noise_name, level)
                    os.makedirs(ber_dir, exist_ok=True)
                    np.savetxt(os.path.join(ber_dir, f"{value}.dat"), list_ber, fmt="%.4e")

                    ncc_dir = os.path.join(base, "results", "ncc", method_name, noise_name, level)
                    os.makedirs(ncc_dir, exist_ok=True)
                    np.savetxt(os.path.join(ncc_dir, f"{value}.dat"), list_ncc, fmt="%.4e")


            # Actualizar barra de progreso
            progress_bar.printProgressBar(idx + 1, len(noises_list),
                prefix='Progress:', suffix='Complete', length=50)

if __name__ == '__main__':
    main()