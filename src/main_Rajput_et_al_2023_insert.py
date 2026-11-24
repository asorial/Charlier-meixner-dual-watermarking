
from helpers.metrics import Metrics
from noises_attaks.noises_class_cv2 import NoisesClass
from scipy.linalg import hessenberg, svd
from scipy.fftpack import dct, idct
from helpers.utils import *
import numpy as np
import pywt
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

    sub_ind = ['10']

    for idx in sub_ind:

        # Directorios de resultados
        dir_results = os.path.join(base, "results/")
        dir_psnr = os.path.join(dir_results, f"psnr/")
        if not os.path.exists(dir_psnr):
            os.makedirs(dir_psnr)
        dir_psnr = os.path.join(dir_psnr, f"{idx}.dat")
        dir_uiqi = os.path.join(dir_results, f"uiqi/")
        if not os.path.exists(dir_uiqi):
            os.makedirs(dir_uiqi)
        dir_uiqi = os.path.join(dir_uiqi, f"{idx}.dat")
        dir_image_fid = os.path.join(dir_results, f"image_fid/")
        if not os.path.exists(dir_image_fid):
            os.makedirs(dir_image_fid)
        dir_image_fid = os.path.join(dir_image_fid, f"{idx}.dat")
        dir_ssim = os.path.join(dir_results, f"ssim/")
        if not os.path.exists(dir_ssim):
            os.makedirs(dir_ssim)
        dir_ssim = os.path.join(dir_ssim, f"{idx}.dat")

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
            save_dir = os.path.join(base, f"watermarked_images/Rajput_{idx}/")
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
                "Rajput_" + idx
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