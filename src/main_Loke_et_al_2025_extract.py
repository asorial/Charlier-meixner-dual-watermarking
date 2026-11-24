
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


alpha_par = 0.05

# === Inserción ===

def embed_watermark(cover_rgb: np.ndarray, watermark: np.ndarray, alpha: float = alpha_par):
    """
    Inserta una marca de agua en una imagen de portada usando DWT-DCT-SVD.

    Args:
        cover_rgb (np.ndarray): La imagen de portada en formato RGB de 512x512x3.
        watermark (np.ndarray): La marca de agua en escala de grises.
        alpha (float): Factor de ganancia para controlar la fuerza de la marca de agua.

    Returns:
        tuple: (watermarked_rgb, Uw, Vw)
    """
    # 1. Separar la imagen en canales de color. El artículo menciona usar los 3, pero el diagrama se enfoca en uno,
    # así que para este ejemplo usaremos el canal G, que es la luminancia.
    R, G, B = cv2.split(cover_rgb)

    # 2. Aplicar un DWT de 1 nivel usando un filtro HAAR en el canal G. El artículo menciona db1, pero Haar es el más común y una buena aproximación.
    coeffs = pywt.dwt2(G.astype(np.float64), 'haar')
    LL, (LH, HL, HH) = coeffs

    # 3. Dividir la sub-banda de frecuencia media (LH en este ejemplo) en bloques de 4x4.
    block_size = 4
    rows, cols = LH.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size

    # 4. Extraer los coeficientes DC después de aplicar DCT a cada bloque.
    dc_matrix = np.zeros((num_blocks_row, num_blocks_col))
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = LH[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dc_matrix[i, j] = dct_block[0, 0]

    # 5. Realizar SVD en la matriz DC.
    Uc, Sc, Vc = svd(dc_matrix)

    # 6. Realizar SVD en la marca de agua.
    Uw, Sw, Vw = svd(watermark.astype(np.float64))

    # 7. Incrustar los valores singulares de la marca de agua en los de la imagen de portada.
    watermark_size = Sw.shape[0]
    Sc_modified = Sc.copy()
    Sc_modified[:watermark_size] = Sc[:watermark_size] + alpha * Sw

    # 8. Reconstruir la matriz DC modificada.
    dc_matrix_modified = Uc @ np.diag(Sc_modified) @ Vc

    # 9. Reconstruir la sub-banda LH con los coeficientes DC modificados.
    LH_modified = LH.copy()
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = LH[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            # Reemplazar solo el coeficiente DC
            dct_block[0, 0] = dc_matrix_modified[i, j]
            # IDCT del bloque modificado
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            LH_modified[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = idct_block

    # 10. Realizar el IDWT para obtener el canal G marcado.
    coeffs_modified = (LL, (LH_modified, HL, HH))
    G_watermarked = pywt.idwt2(coeffs_modified, 'haar')
    G_watermarked = np.clip(G_watermarked, 0, 255).astype(np.uint8)

    # 11. Unir la imagen con marca de agua con los canales originales R y B.
    watermarked_rgb = cv2.merge((R, G_watermarked, B))

    return watermarked_rgb, Uw, Vw

# === Extracción de marca de agua ===

def extract_watermark(watermarked_rgb: np.ndarray, cover_rgb: np.ndarray, Uw: np.ndarray, Vw: np.ndarray, alpha: float = alpha_par) -> np.ndarray:
    """
    Extrae la marca de agua de la imagen marcada usando DWT-DCT-SVD.

    Args:
        watermarked_rgb (np.ndarray): La imagen con la marca de agua.
        cover_rgb (np.ndarray): La imagen de portada original.
        Uw (np.ndarray): La matriz U de la marca de agua original.
        Vw (np.ndarray): La matriz V de la marca de agua original.
        alpha (float): Factor de ganancia utilizado durante la inserción.

    Returns:
        np.ndarray: La marca de agua extraída.
    """
    # 1. Separar las imágenes en canales de color. Usaremos el canal G.
    _, G_watermarked, _ = cv2.split(watermarked_rgb)
    _, G_cover, _ = cv2.split(cover_rgb)

    # 2. Aplicar DWT de 1 nivel en ambas imágenes y obtener las sub-bandas LH.
    coeffs_w = pywt.dwt2(G_watermarked.astype(np.float64), 'haar')
    LH_w = coeffs_w[1][0]

    coeffs_c = pywt.dwt2(G_cover.astype(np.float64), 'haar')
    LH_c = coeffs_c[1][0]

    # 3. Extraer los coeficientes DC de ambas sub-bandas LH.
    block_size = 4
    rows, cols = LH_w.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size

    dc_matrix_w = np.zeros((num_blocks_row, num_blocks_col))
    dc_matrix_c = np.zeros((num_blocks_row, num_blocks_col))

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block_w = LH_w[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block_w = dct(dct(block_w.T, norm='ortho').T, norm='ortho')
            dc_matrix_w[i, j] = dct_block_w[0, 0]

            block_c = LH_c[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block_c = dct(dct(block_c.T, norm='ortho').T, norm='ortho')
            dc_matrix_c[i, j] = dct_block_c[0, 0]

    # 4. Realizar SVD en ambas matrices DC.
    _, Sc_w, _ = svd(dc_matrix_w)
    _, Sc_c, _ = svd(dc_matrix_c)

    # 5. Calcular los valores singulares de la marca de agua extraída.
    watermark_size = Uw.shape[0]
    Sw_extracted = (Sc_w[:watermark_size] - Sc_c[:watermark_size]) / alpha

    # 6. Reconstruir la marca de agua usando las matrices U y V originales de la marca de agua.
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

    sub_ind = ['12']

    for value in sub_ind:

        # Nombre del método
        method_name = f"Loke_{value}"
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