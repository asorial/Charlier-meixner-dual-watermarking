import numpy as np
import imageio.v2 as imageio
import os
from helpers import progress_bar
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
        count = np.zeros(watermark_shape)

        # Procesar por bloques 8x8
        for i in range(0, int(h.real), 8):
            for j in range(0, int(w.real), 8):
                if i + 8 > int(h.real) or j + 8 > int(w.real):
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
                        extracted_watermark[m, n] = diff
                        count[m, n] += 1

        # Promediar y escalar
        extracted_watermark = np.divide(
            extracted_watermark,
            count,
            out=np.zeros_like(extracted_watermark),
            where=(count != 0)
        )
        extracted_watermark /= self.alpha

        # Binarización
        extracted_watermark = (extracted_watermark > 0.5).astype(np.uint8)
        return extracted_watermark.flatten()


def main():
    # Instancia para métricas
    metr = Metrics()

    # Directorios base
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_attacked = os.path.join(base, "watermarked_with_nosises")

    # Directorio de marcas de agua
    dir_watermark = os.path.join(base, "watermark")
    watermark_name = os.listdir(dir_watermark)[0]
    dir_watermark_image = os.path.join(dir_watermark, watermark_name)
    watermark_array = imageio.imread(dir_watermark_image)

    # Procesar marca de agua (igual que en inserción)
    # watermark_array = arnold_transform(watermark_array, 27)
    watermark_flat = watermark_array.flatten()
    watermark_binary = binarize_watermark(watermark_flat)
    watermark_shape = watermark_array.shape

    # Configuraciones de Möbius (deben ser las mismas que en inserción)
    mobius_configs = [
        {'a': 1.0, 'b': 0.0, 'c': 0.0, 'd': 1.0},  # Identidad
        {'a': 0.5, 'b': 0.5, 'c': -0.5, 'd': 0.5}   # Configuración compleja
    ]
    sub_ind = ['5', '6']

    for item, config in enumerate(mobius_configs):
        # Crear instancia de watermarking
        waterk = MobiusDCTWatermarking(**config, alpha=0.1)

        # Nombre del método
        method_name = f"MDCTW_{sub_ind[item]}"
        dir_method = os.path.join(dir_attacked, method_name)

        if not os.path.exists(dir_method):
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
                    attacked_img = imageio.imread(img_path)

                    # Cargar imagen original correspondiente
                    original_img_path = os.path.join(
                        base, "cover_images",
                        image_name.split('_')[0] + ".bmp"  # Asumiendo que las imágenes originales están en cover_images
                    )

                    if not os.path.exists(original_img_path):
                        print(f"Original image not found: {original_img_path}")
                        continue
                    original_img = imageio.imread(original_img_path)

                    # Extraer marca de agua
                    extracted_watermark = waterk.extract(
                        attacked_img, original_img, watermark_shape
                    )
                    assert False, extracted_watermark  # Debugging line
                    # Unbinarizar la marca extraída
                    extracted_watermark = unbinarize_watermark(
                        extracted_watermark
                    )
                    # Convertir a matriz
                    extracted_watermark = np.array(extracted_watermark).reshape(watermark_shape)

                    # Aplicar transformada inversa de Arnold
                    extracted_watermark = arnold_transform(
                        extracted_watermark, 37
                    )

                    # Guardar marca extraída
                    save_dir = os.path.join(base, "watermark_ext_with_noise", method_name, noise_name, level)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, image_name)
                    assert False, extracted_watermark
                    extracted_watermark_img = (extracted_watermark * 255).astype(np.uint8)
                    imageio.imsave(save_path, extracted_watermark_img)

                    # Preparar para métricas
                    extracted_flat = extracted_watermark.flatten()

                    # Binarizar la marca extraída
                    extracted_binary = binarize_watermark(extracted_flat)
                    print(f"Extracwatermark_binary: {watermark_binary[:4096]}")  # Debugging line
                    assert False, extracted_binary[:4096]

                    # Calcular métricas
                    ber = metr.ber(watermark_binary, extracted_binary)
                    ncc = metr.ncc(watermark_array, extracted_watermark_img)

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
                    np.savetxt(os.path.join(ber_dir, f"{img_idx}.dat"), [ber], fmt="%.4e")

                    ncc_dir = os.path.join(base, "results", "ncc", method_name, noise_name, level)
                    os.makedirs(ncc_dir, exist_ok=True)
                    np.savetxt(os.path.join(ncc_dir, f"{img_idx}.dat"), [ncc], fmt="%.4e")

                # Guardar resultados completos por nivel
                # np.savetxt(os.path.join(ber_dir, "all_ber.dat"), list_ber, fmt="%.4e")
                # np.savetxt(os.path.join(ncc_dir, "all_ncc.dat"), list_ncc, fmt="%.4e")

            # Actualizar barra de progreso
            progress_bar.printProgressBar(idx + 1, len(noises_list),
                prefix='Progress:', suffix='Complete', length=50)

if __name__ == '__main__':
    main()