
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


alpha_par = 0.001  # Valor por defecto para alpha

# === Inserción ===

def embed_watermark(cover_rgb: np.ndarray, watermark: np.ndarray, alpha: float = alpha_par):
    """
    Inserta una marca de agua en una imagen de portada usando DWT-SVD, según el artículo.
    Retorna la imagen marcada y las matrices Uw, Vw de la marca de agua original.

    Args:
        cover_rgb (np.ndarray): La imagen de portada en formato RGB de 512x512x3.
        watermark (np.ndarray): La marca de agua binaria de 64x64.
        alpha (float): Factor de ganancia para controlar la fuerza de la marca de agua.

    Returns:
        tuple: (watermarked_rgb, Uw, Vw)
    """
    # 1. Separar la imagen en canales de color. Usaremos el canal azul para el watermarking.
    R, G, B = cv2.split(cover_rgb)

    # Aplicar un DWT de 3 niveles usando un filtro HAAR en el canal B.
    coeffs = pywt.wavedec2(B.astype(np.float64), 'haar', level=3)
    LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

    # 2. Realizar SVD en el sub-banda LL3[cite: 264, 265, 266].
    Uc, Sc, Vc = svd(LL3)

    # 3. Realizar SVD en la marca de agua[cite: 268, 269, 270, 271].
    Uw, Sw, Vw = svd(watermark.astype(np.float64))

    # 4. Incrustar los valores singulares de la marca de agua en los de la imagen de portada[cite: 273, 274, 275].
    # La incrustación se realiza en la sub-banda LL3 que tiene el mismo tamaño que la marca de agua.
    Sc_modified = Sc + alpha * Sw

    # 5. Reconstruir la sub-banda LL3[cite: 277, 278, 279].
    LL3_modified = Uc @ np.diag(Sc_modified) @ Vc

    # Reconstruir los coeficientes DWT.
    coeffs_modified = (LL3_modified, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1))

    # 6. Realizar el IDWT de 3 niveles para obtener la imagen marcada[cite: 281].
    B_watermarked = pywt.waverec2(coeffs_modified, 'haar')
    B_watermarked = np.clip(B_watermarked, 0, 255).astype(np.uint8)

    # Unir la imagen con marca de agua con los canales originales R y G.
    watermarked_rgb = cv2.merge((R, G, B_watermarked))

    return watermarked_rgb, Uw, Vw


# ===================================================

# === Extracción de marca de agua ===

def extract_watermark(watermarked_rgb: np.ndarray, cover_rgb: np.ndarray, Uw: np.ndarray, Vw: np.ndarray, alpha: float = alpha_par) -> np.ndarray:
    """
    Extrae la marca de agua de la imagen marcada.
    Utiliza las matrices Uw y Vw de la marca de agua original para la reconstrucción.

    Args:
        watermarked_rgb (np.ndarray): La imagen con la marca de agua.
        cover_rgb (np.ndarray): La imagen de portada original.
        Uw (np.ndarray): La matriz U de la marca de agua original.
        Vw (np.ndarray): La matriz V de la marca de agua original.
        alpha (float): Factor de ganancia utilizado durante la inserción.

    Returns:
        np.ndarray: La marca de agua extraída.
    """
    # 1. Separar las imágenes en canales de color. Usaremos el canal azul.
    _, _, B_watermarked = cv2.split(watermarked_rgb)
    _, _, B_cover = cv2.split(cover_rgb)

    # 2. Aplicar DWT de 3 niveles en ambas imágenes y obtener las sub-bandas LL3[cite: 284].
    coeffs_w = pywt.wavedec2(B_watermarked.astype(np.float64), 'haar', level=3)
    LL3_w = coeffs_w[0]

    coeffs_c = pywt.wavedec2(B_cover.astype(np.float64), 'haar', level=3)
    LL3_c = coeffs_c[0]

    # 3. Realizar SVD en ambas sub-bandas LL3[cite: 286, 287, 288, 289].
    _, Sw, _ = svd(LL3_w)
    _, Sc, _ = svd(LL3_c)

    # 4. Calcular los valores singulares de la marca de agua extraída[cite: 291, 292, 293].
    Sw_extracted = (Sw - Sc) / alpha

    # 5. Reconstruir la marca de agua usando las matrices U y V originales de la marca de agua[cite: 295, 296].
    watermark_extracted = Uw @ np.diag(Sw_extracted) @ Vw

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

    sub_ind = ['11']

    for value in sub_ind:

        # Nombre del método
        method_name = f"Alomoush_{value}"
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