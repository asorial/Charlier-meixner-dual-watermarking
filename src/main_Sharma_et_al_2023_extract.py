
from helpers import progress_bar
from helpers.metrics import Metrics
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
# EXTRACCIÓN EN CANAL R (sub-banda LH2, bloques 2x2)
# ===================================================

def extract_watermark(watermarked_rgb):
    """
    Extrae la marca de agua desde el canal R usando codificación relacional.
    - watermarked_rgb: imagen RGB con marca
    - return: imagen 64x64 en escala de grises
    """
    R_channel = watermarked_rgb[:, :, 2]

    # DWT hasta nivel 2
    cA1, (cH1, cV1, cD1) = dwt2(R_channel, 'haar')
    cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar')

    bits = []
    h, w = cH2.shape
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = cH2[i:i+2, j:j+2]
            a, b = block[0, 0], block[1, 1]
            bit = 1 if a > b else 0
            bits.append(bit)
            if len(bits) >= 4096:
                break
        if len(bits) >= 4096:
            break

    return bits_to_img(np.array(bits))

# ===================================================
# MAIN
# ===================================================

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

    watermark_flat = watermark_array.flatten()
    watermark_binary = binarize_watermark(watermark_flat)

    sub_ind = ['9']

    for value in sub_ind:

        # Nombre del método
        method_name = f"Sharma_{value}"
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

                    # Extraer marca de agua
                    extracted_watermark = extract_watermark(attacked_img)

                    # Guardar marca extraída
                    save_dir = os.path.join(base, "watermark_ext_with_noise", method_name, noise_name, level)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, image_name)

                    # Save as image
                    cv2.imwrite(save_path, extracted_watermark)

                    # Preparar para métricas
                    extracted_flat = extracted_watermark.flatten()

                    # Binarizar la marca extraída
                    extracted_binary = binarize_watermark(extracted_flat)
                    # print(f"Extracwatermark_binary: {watermark_binary[:100]}")  # Debugging line
                    # assert False, extracted_binary[:100]

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