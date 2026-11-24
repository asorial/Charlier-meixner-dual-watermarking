
from helpers.orthonornal_matrix import orthmatrix
from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.metrics import Metrics
from helpers import progress_bar
import imageio.v2 as imageio
from helpers.utils import *
import imageio as iio
import numpy as np
import os


def main():
    # Instance
    metr = Metrics()
    # Directory
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "watermarked_with_nosises")

    # Watermark directory
    dir_watermark = os.path.join(base, "watermark")
    if not os.path.exists(dir_watermark): os.makedirs(dir_watermark)
    # Watermark arrays
    watermark_name = os.listdir(dir_watermark)
    dir_watermark_image = os.path.join(
        dir_watermark,
        watermark_name[0]
    )
    watermark_array = iio.imread(dir_watermark_image)

    # Watermark in the form of a list
    watermark_as_list = watermark_array.reshape(
            (1, watermark_array.size)
        )[0]

    # Krawtchouk_sobolev Orthogonal matrix
    matrixs = [orthmatrix(4)]

    sub_ind = ['5']

    for item, matrix in enumerate(matrixs):
        # Initial values
        method_name = "KS_"
        method_name += sub_ind[item]
        dir_method_name = os.path.join(dir_covers, method_name)
        noises_directory = os.listdir(dir_method_name)
        noises_directory = sorted(noises_directory, key=lambda x: int(x.split("_")[0]))

        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(noises_directory), prefix='Progress:',
            suffix='Complete', length=50)
        for idx, noise_name in enumerate(noises_directory):
            dir_noise = os.path.join(dir_method_name, noise_name)
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

                # Extracting watermark
                for index, image_name in enumerate(image_names):
                    # Instance
                    waterk = DualWatermarkingScheme(
                        "N;fpr-y7", 150, 62, 0.58, 0.24, matrix, matrix
                    )

                    # Load watermarked images
                    dir_image = os.path.join(
                        dir_level,
                        image_name
                    )
                    watermarked_array = imageio.imread(dir_image)
                    # Extracting the watermark images
                    watermark_array_ext = waterk.extraction(
                        watermarked_array, image_name
                    )
                    # unBinarize the watermark
                    watermark_array_ext = unbinarize_watermark(
                        [
                            int(bit) for bit in watermark_array_ext
                        ]
                    )
                    # Watermark in the form of a matrix
                    watermark_array_ext = np.array(watermark_array_ext).reshape(64, 64)
                    # Applying the Arnold transform
                    watermark_array_ext = arnold_transform(watermark_array_ext, 37)
                    # Creating directory
                    dir_save = os.path.join(
                        base,
                        "watermark_ext_with_noise",
                        dir_level.split("watermarked_with_nosises/")[-1],
                    )
                    if not os.path.exists(dir_save): os.makedirs(dir_save)
                    dir_save = os.path.join(
                        dir_save,
                        dir_image.split("/")[-1],
                    )
                    # Save stego image
                    watermark_array_ext = np.clip(watermark_array_ext, 0, 255).astype(np.uint8)
                    imageio.imsave(dir_save, watermark_array_ext)

                    # Calcular métricas
                    ber, ncc = metr.calculate_ber_ncc(
                        watermark_array, watermark_array_ext
                    )

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
                    np.savetxt(os.path.join(ber_dir, f"{sub_ind[item]}.dat"), list_ber, fmt="%.4e")

                    ncc_dir = os.path.join(base, "results", "ncc", method_name, noise_name, level)
                    os.makedirs(ncc_dir, exist_ok=True)
                    np.savetxt(os.path.join(ncc_dir, f"{sub_ind[item]}.dat"), list_ncc, fmt="%.4e")


            # Actualizar barra de progreso
            progress_bar.printProgressBar(
                idx + 1, len(noises_directory),
                prefix='Progress:', suffix='Complete', length=50)

if __name__ == '__main__':
    main()