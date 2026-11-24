
from helpers import progress_bar
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
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

# ===============================
# Función de extracción de marca
# ===============================

def extract_watermark(watermarked_rgb, alpha=0.05):
    """
    Extrae la marca de agua de la imagen RGB usando solo el canal R.
    - watermarked_rgb: imagen RGB marcada
    """
    # Cargar contexto
    Sc = np.load('src/_Sc.npy')
    Uw = np.load('src/_Uw.npy')
    Vw = np.load('src/_Vw.npy')
    arnold_iter = int(np.load('src/_arnold_iter.npy')[0])

    # Obtener canal R
    wm_r = watermarked_rgb[:, :, 0]

    # DCT + DWT 3 niveles
    dct_wm = dct(dct(wm_r.T, norm='ortho').T, norm='ortho')
    cA1, (cH1, cV1, cD1) = dwt2(dct_wm, 'haar')
    cA2, (cH2, cV2, cD2) = dwt2(cA1, 'haar')
    cA3, (cH3, cV3, cD3) = dwt2(cA2, 'haar')

    # SVD del HH3 extraído
    _, Sc_wm, _ = svd(cD3, full_matrices=False)

    # Extraer Sw = (Sc' - Sc) / alpha
    Sw_extracted = (Sc_wm - Sc) / alpha
    watermark_recovered = np.dot(Uw, np.dot(np.diag(Sw_extracted), Vw))
    watermark_recovered = np.clip(watermark_recovered, 0, 255).astype(np.uint8)

    # Aplicar inversa de Arnold
    watermark_array = inverse_arnold_transform(watermark_recovered, arnold_iter)
    return watermark_array

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

    sub_ind = ['7']

    for value in sub_ind:

        # Nombre del método
        method_name = f"Begum_{value}"
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

                    # Cargar imagen original correspondiente
                    original_img_path = os.path.join(
                        base, "cover_images",
                        image_name.split('_')[0] + ".bmp"  # Asumiendo que las imágenes originales están en cover_images
                    )

                    # Verificar si la imagen original existe
                    if not os.path.exists(original_img_path):
                        print(f"Original image not found: {original_img_path}")
                        continue
                    # original_img = cv2.imread(original_img_path)

                    # _, Uw, Vw = embed_watermark(
                    #     original_img,
                    #     watermark_bin
                    # )

                    # Extraer marca de agua
                    extracted_watermark = extract_watermark(
                        attacked_img, alpha=0.05
                    )

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