
from helpers.metrics import Metrics
from noises_attaks.noises_class_cv2 import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import *
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from pywt import dwt2, idwt2
import numpy as np
import cv2
import os

# ===============================
# Funciones de transformación Arnold
# ===============================

def arnold_transform(image, iterations=1):
    """Aplica la transformación de Arnold a una imagen cuadrada."""
    N = image.shape[0]
    result = image.copy()
    for _ in range(iterations):
        result = np.array([[result[(i + j) % N][(i + 2 * j) % N] for j in range(N)] for i in range(N)])
    return np.array(result)

def inverse_arnold_transform(image, iterations=1):
    """Aplica la transformación de Arnold inversa."""
    N = image.shape[0]
    result = image.copy()
    for _ in range(iterations):
        result = np.array([[result[(2 * i - j) % N][(-i + j) % N] for j in range(N)] for i in range(N)])
    return np.array(result)

# ===============================
# Función de inserción de marca
# ===============================

def embed_watermark(cover_rgb, watermark_gray, alpha=0.05, arnold_iter=50):
    """
    Inserta una watermark en el canal R de una imagen RGB.
    - cover_rgb: imagen RGB (numpy array H x W x 3)
    - watermark_gray: imagen en escala de grises
    - alpha: fuerza del watermark
    - arnold_iter: número de iteraciones Arnold
    """
    # Tomar canal R
    cover_r = cover_rgb[:, :, 0]

    # Redimensionar y cifrar la watermark
    watermark = cv2.resize(watermark_gray, (64, 64))
    watermark_enc = arnold_transform(watermark, arnold_iter)

    # Aplicar DCT + DWT 3 niveles sobre el canal R
    dct_cover = dct(dct(cover_r.T, norm='ortho').T, norm='ortho')
    cA1, (cH1, cV1, cD1) = dwt2(dct_cover, 'haar')
    cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar')
    cA3, (cH3, cV3, cD3) = dwt2(cA2, 'haar')

    # SVD de HH3 y de la watermark
    Uc, Sc, Vc = svd(cD3, full_matrices=False)
    Uw, Sw, Vw = svd(watermark_enc, full_matrices=False)

    # Embebido: Sc' = Sc + alpha * Sw
    Sc_embedded = Sc + alpha * Sw
    cD3_embedded = np.dot(Uc, np.dot(np.diag(Sc_embedded), Vc))

    # Reconstruir canal R modificado
    cA2_rec = idwt2((cA3, (cH3, cV3, cD3_embedded)), 'haar')
    cA1_rec = idwt2((cA2_rec, (cH2, cV2, cD2)), 'haar')
    dct_rec = idwt2((cA1_rec, (cH1, cV1, cD1)), 'haar')
    cover_r_embedded = idct(idct(dct_rec.T, norm='ortho').T, norm='ortho')
    cover_r_embedded = np.clip(cover_r_embedded, 0, 255).astype(np.uint8)

    # Reemplazar canal R en imagen RGB
    watermarked_rgb = cover_rgb.copy()
    watermarked_rgb[:, :, 0] = cover_r_embedded

    # Guardar datos necesarios para extracción
    np.save('src/_Sc.npy', Sc)
    np.save('src/_Uw.npy', Uw)
    np.save('src/_Vw.npy', Vw)
    np.save('src/_arnold_iter.npy', np.array([arnold_iter]))

    return watermarked_rgb

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

    sub_ind = ['7']

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
            watermarked_array = embed_watermark(
                cover_array, watermark, alpha=0.05, arnold_iter=50
            )

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/Begum_{idx}/")
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
                "Begum_" + idx
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