import numpy as np
import imageio.v3 as iio
import os
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import *
import cv2
import pywt


sm = SeparableMoments(orthmatrix(7), orthmatrix(7))

def dct2(block):
    return sm.direct_moments(block)

def idct2(block):
    return sm.inverse_moments(block)

def arnold_transform(image, iterations):
    """Aplica la transformada de Arnold a una imagen."""
    n = image.shape[0]
    scrambled = np.copy(image)
    for _ in range(iterations):
        temp = np.zeros_like(scrambled)
        for i in range(n):
            for j in range(n):
                ni = (i + j) % n
                nj = (i + 2 * j) % n
                temp[ni, nj] = scrambled[i, j]
        scrambled = temp
    return scrambled

def inverse_arnold(image, iterations):
    """Aplica la inversa de la transformada de Arnold."""
    n = image.shape[0]
    descrambled = np.copy(image)
    for _ in range(iterations):
        temp = np.zeros_like(descrambled)
        for i in range(n):
            for j in range(n):
                ni = (2 * i - j) % n
                nj = (-i + j) % n
                temp[ni % n, nj % n] = descrambled[i, j]
        descrambled = temp
    return descrambled

def embed_watermark(cover_image, watermark, k=40, arnold_iterations=10):
    """Inserta una marca de agua en una imagen RGB según el método del artículo."""
    # 1. Separar la imagen RGB en componentes
    R, G, B = cv2.split(cover_image)
    color_components = [R, G, B]
    watermarked_components = []

    # 2. Preparar la marca de agua
    # a. Transformada de Arnold
    scrambled_watermark = arnold_transform(watermark, arnold_iterations)

    # b. Convertir a binario (9216x8)
    watermark_bin = np.unpackbits(scrambled_watermark).reshape(-1, 8)

    # c. Dividir en 3 partes y rellenar
    parts = np.split(watermark_bin, 3)
    watermark_blocks = []
    for part in parts:
        block = np.pad(part.flatten(), (0, 25600 - part.size), 'constant')
        watermark_blocks.append(block.reshape(160, 160))

    # Procesar cada componente de color
    for c_idx, I_c in enumerate(color_components):
        # 3. Aplicar DCT global
        I_c_DCT = cv2.dct(I_c.astype(np.float64))

        # 4. Aplicar DWT (1 nivel, Haar)
        LL, (LH, HL, HH) = pywt.dwt2(I_c_DCT, 'haar')
        bands = [LL, LH, HL, HH]

        # 5. Procesar bloque de marca de agua
        W_c = watermark_blocks[c_idx]
        W_c_DCT = cv2.dct(W_c.astype(np.float64))

        # 6. Dividir en sub-bloques 80x80
        sub_blocks = [
            W_c_DCT[:80, :80],
            W_c_DCT[:80, 80:],
            W_c_DCT[80:, :80],
            W_c_DCT[80:, 80:]
        ]

        # 7. Insertar en bandas DWT
        watermarked_bands = []
        for band, sub_block in zip(bands, sub_blocks):
            # Crear matriz 512x512 con el sub-bloque
            embedded_block = np.zeros_like(band)
            embedded_block[:80, :80] = sub_block
            # Incrustar con factor de escala
            watermarked_band = band + embedded_block * k
            watermarked_bands.append(watermarked_band)

        # 8. Reconstrucción
        LL_w, LH_w, HL_w, HH_w = watermarked_bands
        I_c_DCT_w = pywt.idwt2((LL_w, (LH_w, HL_w, HH_w)), 'haar')
        I_c_w = cv2.idct(I_c_DCT_w)
        watermarked_components.append(I_c_w)

    # 9. Combinar componentes
    watermarked_image = cv2.merge(watermarked_components)
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def main():
    # Instancia para métricas
    metr = Metrics()

    # Directorios
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "cover_images/")
    dir_watermark = os.path.join(base, "watermark")
    dir_watermark_image = os.path.join(dir_watermark, os.listdir(dir_watermark)[0])

    # Leer marca de agua
    watermark_array = iio.imread(dir_watermark_image)

    sub_ind = ['5']

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
            cover_array = iio.imread(cover_path)

            # Insertar marca de agua
            watermarked_array = embed_watermark(
                cover_array, watermark_array, k=40, arnold_iterations=10
            )

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/dct-dwt_{idx}/")
            os.makedirs(save_dir, exist_ok=True)
            iio.imwrite(os.path.join(save_dir, image_name), watermarked_array)

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
                "dct-dwt_" + idx
            )

            index += 1

            noise.gaussian_noise([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

            noise.minimum_filter_noise(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                )

            noise.gaussian_laplace(
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                )

            noise.percentile_filter(
                    [
                        0.1, 0.2, 0.3, 0.4,
                        0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                    ]
                )

            noise.uniform_filter([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

            noise.fourier_ellipsoid_filter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            noise.fourier_gaussian_filter(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                )

            noise.fourier_uniform_filter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

            noise.salt_pepper_noise(
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                )

            noise.cropping_noise([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            noise.median_filter_noise([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            noise.scaling_noise(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2]
            )

            noise.rotation_noise(
                [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75]
            )
            # noise.jpeg_noise([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50])

if __name__ == '__main__':
    main()