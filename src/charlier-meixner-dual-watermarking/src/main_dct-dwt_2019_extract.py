import numpy as np
import imageio.v2 as imageio
import imageio.v3 as iio
import os
from helpers import progress_bar
from helpers.metrics import Metrics
from noises_attaks.noises_class import NoisesClass
from moments_matrix.moments_class import SeparableMoments
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import *
import cv2
import pywt

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

def extract_watermark(original_image, watermarked_image, k=40, arnold_iterations=10, watermark_size=96):
    """Extrae la marca de agua de una imagen según el método del artículo."""
    # 1. Separar componentes
    R_o, G_o, B_o = cv2.split(original_image)
    R_w, G_w, B_w = cv2.split(watermarked_image)
    orig_components = [R_o, G_o, B_o]
    wm_components = [R_w, G_w, B_w]
    extracted_parts = []
    
    # Tamaño de datos útiles por canal (96x96x8/3 = 24576 bits)
    useful_bits_per_channel = (watermark_size * watermark_size * 8) // 3
    
    for orig, wm in zip(orig_components, wm_components):
        # 2. Aplicar DCT a ambas imágenes
        orig_DCT = cv2.dct(orig.astype(np.float64))
        wm_DCT = cv2.dct(wm.astype(np.float64))
        
        # 3. Calcular bandas DWT para ambas imágenes
        orig_LL, (orig_LH, orig_HL, orig_HH) = pywt.dwt2(orig_DCT, 'haar')
        wm_LL, (wm_LH, wm_HL, wm_HH) = pywt.dwt2(wm_DCT, 'haar')
        orig_bands = [orig_LL, orig_LH, orig_HL, orig_HH]
        wm_bands = [wm_LL, wm_LH, wm_HL, wm_HH]
        
        # 4. Extraer marcas de todas las bandas
        extracted_blocks = []
        for orig_band, wm_band in zip(orig_bands, wm_bands):
            W = (wm_band - orig_band) / k
            extracted_blocks.append(W[:80, :80])  # Tomar esquina 80x80
        
        # 5. Reconstruir bloque DCT (160x160)
        top = np.hstack((extracted_blocks[0], extracted_blocks[1]))
        bottom = np.hstack((extracted_blocks[2], extracted_blocks[3]))
        W_DCT = np.vstack((top, bottom))
        
        # 6. Aplicar IDCT
        W_block = cv2.idct(W_DCT)
        
        # 7. Tomar solo datos útiles y binarizar
        flat_block = W_block.flatten()[:useful_bits_per_channel]
        binary_block = (flat_block > 0.5).astype(np.uint8)
        extracted_parts.append(binary_block)
    
    # 8. Combinar y empaquetar bits
    combined_bits = np.concatenate(extracted_parts)
    binary_watermark = np.packbits(combined_bits)
    
    # 9. Reformar a imagen 96x96
    extracted_watermark = binary_watermark[:watermark_size*watermark_size].reshape(watermark_size, watermark_size)
    
    # 10. Aplicar inversa de Arnold
    return inverse_arnold(extracted_watermark, arnold_iterations)

# def extract_watermark(original_image, watermarked_image, k=40, arnold_iterations=10, watermark_size=96):
#     """Extrae la marca de agua de una imagen según el método del artículo."""
#     # 1. Separar componentes
#     R_o, G_o, B_o = cv2.split(original_image)
#     R_w, G_w, B_w = cv2.split(watermarked_image)
#     orig_components = [R_o, G_o, B_o]
#     wm_components = [R_w, G_w, B_w]
#     extracted_parts = []

#     for orig, wm in zip(orig_components, wm_components):
#         # 2. Aplicar DCT + DWT a ambas imágenes
#         orig_DCT = cv2.dct(orig.astype(np.float64))
#         wm_DCT = cv2.dct(wm.astype(np.float64))

#         orig_LL, _ = pywt.dwt2(orig_DCT, 'haar')
#         wm_LL, wm_coeffs = pywt.dwt2(wm_DCT, 'haar')
#         wm_LH, wm_HL, wm_HH = wm_coeffs

#         # 3. Extraer marcas de todas las bandas
#         extracted_blocks = []
#         for band in [wm_LL, wm_LH, wm_HL, wm_HH]:
#             # Extraer componente
#             orig_band, _ = pywt.dwt2(orig_DCT, 'haar')
#             W = (band - orig_band) / k
#             extracted_blocks.append(W[:80, :80])

#         # 4. Reconstruir bloque DCT
#         W_DCT = np.vstack((
#             np.hstack((extracted_blocks[0], extracted_blocks[1])),
#             np.hstack((extracted_blocks[2], extracted_blocks[3]))
#         ))

#         # 5. Aplicar IDCT
#         W_block = cv2.idct(W_DCT)
#         extracted_parts.append(W_block[:3072].flatten())

#     # 6. Combinar y descrambling
#     combined_bits = np.concatenate(extracted_parts)
#     binary_watermark = np.packbits(combined_bits[:watermark_size*watermark_size*8])
#     extracted_watermark = binary_watermark[:watermark_size*watermark_size].reshape(watermark_size, watermark_size)
#     return inverse_arnold(extracted_watermark, arnold_iterations)

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

    sub_ind = ['5']

    for value in sub_ind:


        # Nombre del método
        method_name = f"dct-dwt_{value}"
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
                    attacked_img = imageio.imread(img_path)

                    # Cargar imagen original correspondiente
                    original_img_path = os.path.join(
                        base, "cover_images",
                        image_name.split('_')[0] + ".bmp"  # Asumiendo que las imágenes originales están en cover_images
                    )
                    # Verificar si la imagen original existe
                    if not os.path.exists(original_img_path):
                        print(f"Original image not found: {original_img_path}")
                        continue
                    original_img = imageio.imread(original_img_path)

                    # Extraer marca de agua
                    extracted_watermark = extract_watermark(
                        attacked_img,
                        original_img,
                        k=40,
                        arnold_iterations=10,
                        watermark_size=96
                    )
                    assert False, extracted_watermark.flatten()[:1000]  # Debugging line

                    # Guardar marca extraída
                    save_dir = os.path.join(base, "watermark_ext_with_noise", method_name, noise_name, level)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, image_name)
                    extracted_watermark_img = extracted_watermark * 255
                    extracted_watermark_img = extracted_watermark_img.astype(np.uint8)
                    # Save as image
                    imageio.imsave(save_path, extracted_watermark_img)
                    iio.imwrite(save_path, extracted_watermark)

                    # Preparar para métricas
                    extracted_flat = extracted_watermark.flatten()

                    # Binarizar la marca extraída
                    extracted_binary = binarize_watermark(extracted_flat)
                    # print(f"Extracwatermark_binary: {watermark_binary[:100]}")  # Debugging line
                    # assert False, extracted_binary[:100]

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