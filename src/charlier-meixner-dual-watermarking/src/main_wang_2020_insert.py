
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import *
from skimage.color import rgb2ycbcr, ycbcr2rgb
from sklearn.preprocessing import normalize
from scipy.fftpack import dct, idct
import imageio.v3 as iio
import numpy as np
import cv2
import os

# =============================
# CONFIGURACIÓN
# =============================
BLOCK_SIZE = 8
WM_SIZE = 64  # Marca de agua binaria 64x64
MU = 255  # Factor de escalado logarítmico
DELTA = 1 / 32  # Paso de cuantización
np.random.seed(42)  # Para reproducibilidad

# Vector de proyección aleatoria normalizado
v = normalize(np.random.randn(8, 1), axis=0, norm='l2').flatten()

# =============================
# FUNCIONES AUXILIARES
# =============================

def block_dct(img):
    """Divide en bloques y aplica DCT."""
    h, w = img.shape
    return [
        dct(dct(img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].T, norm='ortho').T, norm='ortho')
        for i in range(0, h, BLOCK_SIZE)
        for j in range(0, w, BLOCK_SIZE)
    ]

def block_idct(blocks, shape):
    """Reconstruye imagen desde bloques DCT inversa."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block = idct(idct(blocks[idx].T, norm='ortho').T, norm='ortho')
            img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = block
            idx += 1
    return img

def embed_watermark(Y, watermark_bin, mu=MU, delta=DELTA):
    """Inserta marca de agua en el canal Y."""
    blocks = block_dct(Y)
    wm_idx = 0
    wm_blocks = []

    for block in blocks:
        if wm_idx >= len(watermark_bin):
            wm_blocks.append(block)
            continue

        # Elegimos 8 coeficientes: [1,2,3,5,6,7,8,10] posiciones de zigzag comunes
        coeffs_idx = [(0,1), (1,0), (1,1), (0,2), (2,0), (1,2), (2,1), (0,3)]
        x_vec = np.array([block[i, j] for i, j in coeffs_idx])

        # Proyección logarítmica
        x_proj = np.dot(x_vec, v)
        arg = 1 + mu * x_proj / 100  # 100: valor empírico estable
        if arg <= 0:
            wm_blocks.append(block)
            continue
        y = np.log(arg) / np.log(1 + mu)

        # Cuantización con dither
        w_bit = watermark_bin[wm_idx]
        d_m = 0.5 if w_bit == 0 else -0.5
        y_q = delta * round((y + d_m) / delta) - d_m

        # Proyección inversa
        x_q_proj = (np.exp(y_q * np.log(1 + mu)) - 1) * 100 / mu
        x_w = x_vec + (x_q_proj - x_proj) * v

        # Sustituimos los coeficientes modificados
        block_mod = block.copy()
        for k, (i, j) in enumerate(coeffs_idx):
            block_mod[i, j] = x_w[k]
        wm_blocks.append(block_mod)
        wm_idx += 1

    return block_idct(wm_blocks, Y.shape), wm_blocks

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
    wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # Redimensionamos la marca binaria a 64x64
    wm_bin = cv2.resize(wm, (WM_SIZE, WM_SIZE), interpolation=cv2.INTER_NEAREST)
    wm_bin = (wm_bin > 127).astype(np.uint8).flatten()

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
            cover_array = cv2.imread(cover_path)

            # Insertar marca de agua

            watermarked_array = embed_watermark(
                cover_array, watermark_bits
            )

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/wang_{idx}/")
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
                "wang_" + idx
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