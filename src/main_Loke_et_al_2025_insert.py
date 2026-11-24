
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from numpy.linalg import svd
from scipy.fftpack import dct, idct
from helpers.utils import *
import numpy as np
import pywt
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

# ===================================================
# Funciones auxiliares
# ===================================================

# Función auxiliar para generar listas con n niveles entre un rango
def levels(start, end, n):
    return list(np.linspace(start, end, n))

def levels_int(start, end, n):
    return list(map(int, np.linspace(start, end, n)))

# ===================================================
# MAIN
# ===================================================

def main():
    # Instancia para métricas
    metr = Metrics()

    # Directorios
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "cover_images/")
    watermark_path = os.path.join(base, "watermark")
    watermark_path = os.path.join(
        watermark_path, os.listdir(watermark_path)[0]
    )

    # Leer marca de agua
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    watermark_bin = (np.array(watermark) > 127).astype(np.uint8)

    sub_ind = ['12']

    for idx in sub_ind:

        # Directorios de resultados
        dir_results = os.path.join(base, "results/")
        dir_psnr = os.path.join(dir_results, f"psnr/")
        if not os.path.exists(dir_psnr):
            os.makedirs(dir_psnr)
        dir_psnr = os.path.join(dir_results, f"{idx}.dat")
        dir_uiqi = os.path.join(dir_results, f"uiqi/")
        if not os.path.exists(dir_uiqi):
            os.makedirs(dir_uiqi)
        dir_uiqi = os.path.join(dir_results, f"{idx}.dat")
        dir_image_fid = os.path.join(dir_results, f"image_fid/")
        if not os.path.exists(dir_image_fid):
            os.makedirs(dir_image_fid)
        dir_image_fid = os.path.join(dir_results, f"{idx}.dat")
        dir_ssim = os.path.join(dir_results, f"ssim/")
        if not os.path.exists(dir_ssim):
            os.makedirs(dir_ssim)
        dir_ssim = os.path.join(dir_results, f"{idx}.dat")

        # Inicializar métricas
        M_psnr, M_uiqi, M_image_fid, M_ssim = [], [], [], []

        index = 0

        # Procesar cada imagen de cubierta
        for image_name in os.listdir(dir_covers):
            cover_path = os.path.join(dir_covers, image_name)
            cover_array = cv2.imread(cover_path)

            # Insertar marca de agua
            watermarked_array, _, _ = embed_watermark(
                cover_array, watermark_bin
            )

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/Loke_{idx}/")
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, image_name), watermarked_array)

            # Calcular métricas
            psnr_value = metr.psnr(cover_array, watermarked_array)
            M_psnr.append(psnr_value)
            print(" ")
            print("PSNR: ", psnr_value)
            print(" ")

            uiqi_value = metr.uiqi(cover_array, watermarked_array)
            M_uiqi.append(uiqi_value)
            print(" ")
            print("UIQI: ", uiqi_value)
            print(" ")

            image_fid = metr.image_fid(cover_array, watermarked_array)
            M_image_fid.append(image_fid)
            print(" ")
            print("Image FID: ", image_fid)
            print(" ")

            ssim_value = metr.ssim(cover_array, watermarked_array)
            M_ssim.append(ssim_value)
            print(" ")
            print("SSIM: ", ssim_value)
            print(" ")

            # Guardar métricas
            np.savetxt(dir_psnr, M_psnr, fmt="%.4e")
            np.savetxt(dir_uiqi, M_uiqi, fmt="%.4e")
            np.savetxt(dir_image_fid, M_image_fid, fmt="%.4e")
            np.savetxt(dir_ssim, M_ssim, fmt="%.4e")

            # instance
            noise = NoisesClass(
                watermarked_array,
                str(index),
                image_name.split(".")[0],
                "Loke_" + idx
            )

            index += 1

            noise.gaussian_noise([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])

            noise.minimum_filter_noise(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                )

            noise.gaussian_laplace(
                    [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
                )

            noise.percentile_filter([-20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])

            noise.uniform_filter([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

            noise.fourier_ellipsoid_filter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

            noise.fourier_gaussian_filter(
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
                )

            noise.fourier_uniform_filter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

            noise.salt_pepper_noise(
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
                )

            noise.cropping_noise([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

            # noise.median_filter_noise([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            noise.scaling_noise(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            )

            noise.rotation_noise(
                [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]
            )
            noise.jpeg_noise([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

        # Saving results
        np.savetxt(dir_psnr, M_psnr, fmt="%.4e")
        np.savetxt(dir_uiqi, M_uiqi, fmt="%.4e")
        np.savetxt(dir_image_fid, M_image_fid, fmt="%.4e")
        np.savetxt(dir_ssim, M_ssim, fmt="%.4e")


if __name__ == '__main__':
    main()