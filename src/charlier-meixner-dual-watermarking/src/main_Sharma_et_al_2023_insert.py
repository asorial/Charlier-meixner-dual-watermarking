
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

# ===================================================
# UTILIDADES
# ===================================================

def img_to_bits(img):
    """Convierte una imagen en escala de grises 64x64 a un vector de bits (0/1)."""
    img = cv2.resize(img, (64, 64))
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.flatten()

def bits_to_img(bits):
    """Convierte un vector de bits a una imagen de 64x64 en escala de grises."""
    return (np.reshape(bits[:4096], (64, 64)) * 255).astype(np.uint8)

# ===================================================
# INSERCIÓN EN CANAL R (sub-banda LH2, bloques 2x2)
# ===================================================

def embed_watermark(cover_rgb, watermark_gray):
    """
    Inserta una marca de agua en el canal R usando codificación relacional en LH2.
    - cover_rgb: imagen RGB (np.array HxWx3)
    - watermark_gray: marca 64x64 en escala de grises
    - return: imagen RGB con marca embebida
    """
    # Obtener canal R (OpenCV usa BGR → canal R es el índice 2)
    R_channel = cover_rgb[:, :, 2]

    # DWT hasta nivel 2 en canal R
    cA1, (cH1, cV1, cD1) = dwt2(R_channel, 'haar')
    cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar')  # LH2 = cH2

    # Preparar bits de la marca
    wm_bits = img_to_bits(watermark_gray)

    # Insertar bits usando bloques 2x2: relación entre block[0,0] y block[1,1]
    embedded = cH2.copy()
    h, w = embedded.shape
    idx = 0
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            if idx >= len(wm_bits):
                break
            block = embedded[i:i+2, j:j+2].copy()
            a, b = block[0, 0], block[1, 1]

            if wm_bits[idx] == 1 and a <= b:
                block[0, 0] = b + 2
            elif wm_bits[idx] == 0 and a >= b:
                block[0, 0] = b - 2

            embedded[i:i+2, j:j+2] = block
            idx += 1

    # Reconstrucción inversa
    cA1r = idwt2((cA2, (embedded, cV2, cD2)), 'haar')
    Rr = idwt2((cA1r, (cH1, cV1, cD1)), 'haar')

    # Clipping y reensamblado
    R_final = np.clip(Rr, 0, 255).astype(np.uint8)
    result = cover_rgb.copy()
    result[:, :, 2] = R_final
    return result

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

    sub_ind = ['9']

    for idx in sub_ind:

        # Directorios de resultados
        dir_results = os.path.join(base, "results/")
        dir_psnr = os.path.join(dir_results, f"psnr/{idx}.dat")
        dir_uiqi = os.path.join(dir_results, f"uiqi/{idx}.dat")
        dir_image_fid = os.path.join(dir_results, f"image_fid/{idx}.dat")
        dir_ssim = os.path.join(dir_results, f"ssim/{idx}.dat")

        # Inicializar métricas
        M_psnr, M_uiqi, M_image_fid, M_ssim = [], [], [], []

        index = 0

        # Procesar cada imagen de cubierta
        for image_name in os.listdir(dir_covers):
            cover_path = os.path.join(dir_covers, image_name)
            cover_array = cv2.imread(cover_path)

            # Insertar marca de agua

            watermarked_array = embed_watermark(
                cover_array, watermark
            )

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/Sharma_{idx}/")
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
                "Sharma_" + idx
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