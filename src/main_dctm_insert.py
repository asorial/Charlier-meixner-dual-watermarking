import numpy as np
import imageio.v3 as iio
import os
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import *
import cv2

sm = SeparableMoments(orthmatrix(7), orthmatrix(7))

def dct2(block):
    return sm.direct_moments(block)

def idct2(block):
    return sm.inverse_moments(block)

# Conversiones de color
def rgb2ycbcr(im):
    xform = np.array([[0.299, 0.587, 0.114],
                      [-0.169, -0.331, 0.5],
                      [0.5, -0.419, -0.081]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return ycbcr

def ycbcr2rgb(ycbcr):
    ycbcr = ycbcr.copy()
    ycbcr[:,:,[1,2]] -= 128
    xform = np.array([[1, 0, 1.402],
                      [1, -0.344, -0.714],
                      [1, 1.772, 0]])
    rgb = ycbcr.dot(xform.T)
    return np.clip(rgb, 0, 255)

# Transformada de Möbius (coordenadas complejas)
def mobius_transform(x, y, a, b, c, d):
    z = complex(x, y)
    w = (a * z + b) / (c * z + d)
    return w.real, w.imag

# Proceso de incrustación
def embed_watermark(host, watermark, alpha, mobius_params):
    a, b, c, d = mobius_params
    # Convertir a YCbCr
    ycbcr = rgb2ycbcr(host)
    Y = ycbcr[:,:,0].copy()
    
    # Aplicar DCT a todo el canal Y
    dct_Y = np.zeros_like(Y, dtype=float)
    h, w = Y.shape
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = Y[i:i+8, j:j+8]
            dct_block = dct2(block.astype(float))
            dct_Y[i:i+8, j:j+8] = dct_block
    
    # Incrustar watermark usando transformada de Möbius
    watermarked_dct = dct_Y.copy()
    wm_h, wm_w = watermark.shape
    for x in range(wm_h):
        for y in range(wm_w):
            u, v = mobius_transform(x, y, a, b, c, d)
            u, v = int(round(u)), int(round(v))
            if 0 <= u < h and 0 <= v < w:
                watermarked_dct[u, v] += alpha * watermark[x, y]
    
    # Aplicar IDCT
    watermarked_Y = np.zeros_like(Y, dtype=float)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = watermarked_dct[i:i+8, j:j+8]
            spatial_block = idct2(block)
            watermarked_Y[i:i+8, j:j+8] = spatial_block
    
    # Recortar y actualizar
    watermarked_Y = np.clip(watermarked_Y, 0, 255)
    ycbcr[:,:,0] = watermarked_Y
    watermarked_rgb = ycbcr2rgb(ycbcr)
    return np.clip(watermarked_rgb, 0, 255).astype(np.uint8)

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
    watermark_shape = watermark_array.shape
    watermark_flat = watermark_array.reshape((1, watermark_array.size))[0]
    watermark_binary = binarize_watermark_int(watermark_flat)
    watermark = np.array(watermark_binary).reshape(watermark_shape)

    # Configuraciones de parámetros de Möbius
    mobius_configs = [
        (1.0, 0.0, 0.0, 1.0),  # Identidad
        # (0.8, 0.2, -0.1, 0.9)
    ]
    alpha = 0.15
    sub_ind = ['5', '6', '7']

    for idx, mobius_params in enumerate(mobius_configs):

        # Directorios de resultados
        dir_results = os.path.join(base, "results/")
        dir_psnr = os.path.join(dir_results, f"psnr/{sub_ind[idx]}.dat")
        dir_uiqi = os.path.join(dir_results, f"uiqi/{sub_ind[idx]}.dat")
        dir_image_fid = os.path.join(dir_results, f"image_fid/{sub_ind[idx]}.dat")
        dir_ssim = os.path.join(dir_results, f"ssim/{sub_ind[idx]}.dat")

        # Inicializar métricas
        M_psnr, M_uiqi, M_image_fid, M_ssim = [], [], [], []

        index = 0

        # Procesar cada imagen de cubierta
        for image_name in os.listdir(dir_covers):
            cover_path = os.path.join(dir_covers, image_name)
            cover_array = iio.imread(cover_path)

            # Insertar marca de agua
            watermarked_array = embed_watermark(
                cover_array, watermark, alpha, mobius_params
            )

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/dctm_{sub_ind[idx]}/")
            os.makedirs(save_dir, exist_ok=True)
            iio.imwrite(os.path.join(save_dir, image_name), watermarked_array)

            # Calcular métricas
            psnr_value = metr.psnr(cover_array, watermarked_array)
            M_psnr.append(psnr_value)
            print(" ")
            print("PSNR: ", M_psnr[index])
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
                "dctm_" + sub_ind[idx]
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