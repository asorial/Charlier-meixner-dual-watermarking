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

# Proceso de extracción
def extract_watermark(watermarked, original, wm_shape, alpha, mobius_params):
    a, b, c, d = mobius_params
    # Convertir a YCbCr
    wm_ycbcr = rgb2ycbcr(watermarked)
    orig_ycbcr = rgb2ycbcr(original)
    
    # Obtener canales Y
    wm_Y = wm_ycbcr[:,:,0]
    orig_Y = orig_ycbcr[:,:,0]
    h, w = wm_Y.shape
    
    # Aplicar DCT a ambas imágenes
    wm_dct = np.zeros_like(wm_Y, dtype=float)
    orig_dct = np.zeros_like(orig_Y, dtype=float)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            wm_block = wm_Y[i:i+8, j:j+8]
            orig_block = orig_Y[i:i+8, j:j+8]
            wm_dct[i:i+8, j:j+8] = dct2(wm_block.astype(float))
            orig_dct[i:i+8, j:j+8] = dct2(orig_block.astype(float))
    
    # Extraer watermark
    watermark = np.zeros(wm_shape)
    wm_h, wm_w = wm_shape
    for x in range(wm_h):
        for y in range(wm_w):
            u, v = mobius_transform(x, y, a, b, c, d)
            u, v = int(round(u)), int(round(v))
            if 0 <= u < h and 0 <= v < w:
                watermark[x, y] = (wm_dct[u, v] - orig_dct[u, v]) / alpha
    return watermark

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
        (1.0, 0.0, 0.0, 1.0),  # Identidad
        # (0.8, 0.2, -0.1, 0.9)
    ]
    alpha = 0.15
    sub_ind = ['5', '6', '7']

    for item, mobius_params in enumerate(mobius_configs):

        # Nombre del método
        method_name = f"dctm_{sub_ind[item]}"
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
                    extracted_watermark = extract_watermark(
                        attacked_img, original_img, watermark_shape, alpha, mobius_params
                    )
                    # assert False, extracted_watermark  # Debugging line
                    # # Unbinarizar la marca extraída
                    # extracted_watermark = unbinarize_watermark(
                    #     extracted_watermark
                    # )
                    # # Convertir a matriz
                    # extracted_watermark = np.array(extracted_watermark).reshape(watermark_shape)

                    # # Aplicar transformada inversa de Arnold
                    # extracted_watermark = arnold_transform(
                    #     extracted_watermark, 37
                    # )

                    # Guardar marca extraída
                    save_dir = os.path.join(base, "watermark_ext_with_noise", method_name, noise_name, level)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, image_name)
                    # assert False, extracted_watermark
                    # extracted_watermark_img = (extracted_watermark * 255).astype(np.uint8)
                    # imageio.imsave(save_path, extracted_watermark_img)
                    # iio.imwrite(save_path, extracted_watermark)

                    # Preparar para métricas
                    extracted_flat = extracted_watermark.flatten()

                    # Binarizar la marca extraída
                    extracted_binary = binarize_watermark(extracted_flat)
                    print(f"Extracwatermark_binary: {watermark_binary[:100]}")  # Debugging line
                    assert False, extracted_binary[:100]

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