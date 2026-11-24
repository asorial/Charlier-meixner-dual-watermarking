import numpy as np
import imageio.v3 as iio
import os
from scipy.fftpack import dct, idct
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import *

class MobiusDCTWatermarking:
    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, alpha=1.0):
        self.a = complex(a)
        self.b = complex(b)
        self.c = complex(c)
        self.d = complex(d)
        self.alpha = alpha
        self.sm = SeparableMoments(orthmatrix(7), orthmatrix(7))

    def mobius_transform(self, z):
        numerator = self.a * z + self.b
        denominator = self.c * z + self.d
        if denominator == 0:
            return z
        return numerator / denominator

    def inverse_mobius_transform(self, w):
        numerator = self.d * w - self.b
        denominator = -self.c * w + self.a
        if denominator == 0:
            return w
        return numerator / denominator

    def dct2(self, block):
        return self.sm.direct_moments(block)

    def idct2(self, block):
        return self.sm.inverse_moments(block)

    def rgb_to_ycbcr(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return np.stack([y, cb, cr], axis=2)

    def ycbcr_to_rgb(self, ycbcr):
        y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)

    def insert(self, cover_array, watermark_flat, watermark_shape):
        # Convertir a YCbCr y extraer canal Y
        ycbcr = self.rgb_to_ycbcr(cover_array)
        y_channel = ycbcr[:, :, 0].astype(np.float64)

        # Dimensiones
        h, w = y_channel.shape
        M, N = watermark_shape
        watermark = np.array(watermark_flat).reshape(watermark_shape)

        # Procesar por bloques 8x8
        for i in range(0, int(h.real), 8):
            for j in range(0, int(w.real), 8):
                if i + 8 > int(h.real) or j + 8 > int(w.real):
                    continue

                # Extraer bloque Y
                block = y_channel[i:i+8, j:j+8]

                # Aplicar DCT
                dct_block = self.dct2(block)

                # Crear bloque modificado
                modified_dct = dct_block.copy()

                # Aplicar transformación de Möbius a cada píxel de la marca
                for m in range(M):
                    for n in range(N):
                        # Coordenada original en la marca
                        z = complex(m, n)
                        w = self.mobius_transform(z)

                        # Mapear al dominio DCT (8x8)
                        x = int(np.real(w)) % 8
                        y = int(np.imag(w)) % 8

                        # Modificar coeficiente DCT
                        modified_dct[x, y] += self.alpha * watermark[m, n]

                # Aplicar IDCT
                modified_block = self.idct2(modified_dct)
                y_channel[i:i+8, j:j+8] = modified_block

        # Combinar canales y convertir a RGB
        ycbcr[:, :, 0] = np.clip(y_channel, 0, 255)
        return self.ycbcr_to_rgb(ycbcr)

    def extract(self, watermarked_array, original_array, watermark_shape):
        # Convertir a YCbCr y extraer canal Y
        ycbcr_w = self.rgb_to_ycbcr(watermarked_array)
        y_w = ycbcr_w[:, :, 0].astype(np.float64)

        ycbcr_o = self.rgb_to_ycbcr(original_array)
        y_o = ycbcr_o[:, :, 0].astype(np.float64)

        # Dimensiones
        h, w = y_w.shape
        M, N = watermark_shape
        extracted_watermark = np.zeros(watermark_shape)

        # Procesar por bloques 8x8
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                if i + 8 > h or j + 8 > w:
                    continue

                # Extraer bloques
                block_w = y_w[i:i+8, j:j+8]
                block_o = y_o[i:i+8, j:j+8]

                # Aplicar DCT
                dct_w = self.dct2(block_w)
                dct_o = self.dct2(block_o)

                # Recuperar marca
                for m in range(M):
                    for n in range(N):
                        z = complex(m, n)
                        w = self.mobius_transform(z)

                        # Mapear al dominio DCT
                        x = int(np.real(w)) % 8
                        y = int(np.imag(w)) % 8

                        # Extraer valor de la marca
                        diff = dct_w[x, y] - dct_o[x, y]
                        extracted_watermark[m, n] = diff / self.alpha

        # Binarizar y aplanar la marca
        extracted_watermark = (extracted_watermark > 0.5).astype(np.uint8)
        return extracted_watermark.flatten()


def main():
    # Instancia para métricas
    metr = Metrics()

    # Directorios
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "cover_images/")
    dir_watermark = os.path.join(base, "watermark")
    dir_watermark_image = os.path.join(dir_watermark, os.listdir(dir_watermark)[0])

    # Leer marca de agua y aplicar transformada de Arnold
    watermark_array = iio.imread(dir_watermark_image)
    watermark_array = arnold_transform(watermark_array, 27)
    watermark_flat = watermark_array.reshape((1, watermark_array.size))[0]
    watermark_flat = binarize_watermark_int(watermark_flat)
    watermark_shape = watermark_array.shape

    # Configuraciones de parámetros de Möbius
    mobius_configs = [
        (1.0, 0.0, 0.0, 1.0),  # Identidad
        (0.5, 0.5, -0.5, 0.5)   # Configuración compleja
    ]
    sub_ind = ['5', '6']

    for idx, config in enumerate(mobius_configs):
        a, b, c, d = config
        waterk = MobiusDCTWatermarking(a, b, c, d, alpha=0.1)

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
            watermarked_array = waterk.insert(cover_array, watermark_flat, watermark_shape)

            # Guardar imagen marcada
            save_dir = os.path.join(base, f"watermarked_images/MDCT_{sub_ind[idx]}/")
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
                "MDCTW_" + sub_ind[idx]
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