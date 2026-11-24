from orthogonal_matrix.orthog_matrix_charlier_sobolev_i_mass import OrthogonalMatrixCharlierSobolevTypePolynomialsIMass
from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.orthonornal_matrix import orthmatrix
from noises_attaks.noises_class import NoisesClass
from helpers.metrics import Metrics
from helpers.utils import *
import imageio.v3 as iio
import numpy as np
import os

from mpmath import *

mp.dps = 25; mp.pretty = True


def main():
    # Instance
    metr = Metrics()
    # Directory
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "cover_images/")
    if not os.path.exists(dir_covers):
        os.makedirs(dir_covers)
    # Cover arrays
    image_names = os.listdir(dir_covers)

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

    # Applying the Arnold transform
    watermark_array = arnold_transform(watermark_array, 27)

    # Watermark in the form of a list
    watermark_as_list = watermark_array.reshape(
            (1, watermark_array.size)
        )[0]

    # Binarize the watermark
    watermark_as_list = binarize_watermark(watermark_as_list)

    # Charlier_sobolev Orthogonal matrix
    matrixs = [orthmatrix(0), orthmatrix(1)]

    sub_ind = ['1', '2']

    # Creating directory
    dir_results = os.path.join(base, "results")

    for item, matrix in enumerate(matrixs):
        # Creating directory
        dir_psnr = os.path.join(
            dir_results, "psnr/"
        )
        if not os.path.exists(dir_psnr):
            os.makedirs(dir_psnr)
        dir_psnr += sub_ind[item]
        dir_psnr = "%s.dat" % dir_psnr

        # Creating directory
        dir_uiqi = os.path.join(
            dir_results, "uiqi/"
        )
        if not os.path.exists(dir_uiqi):
            os.makedirs(dir_uiqi)
        dir_uiqi += sub_ind[item]
        dir_uiqi = "%s.dat" % dir_uiqi

        # Creating directory
        dir_image_fid = os.path.join(
            dir_results, "image_fid/"
        )
        if not os.path.exists(dir_image_fid):
            os.makedirs(dir_image_fid)
        dir_image_fid += sub_ind[item]
        dir_image_fid = "%s.dat" % dir_image_fid

        # Creating directory
        dir_ssim = os.path.join(
            dir_results, "ssim/"
        )
        if not os.path.exists(dir_ssim):
            os.makedirs(dir_ssim)
        dir_ssim += sub_ind[item]
        dir_ssim = "%s.dat" % dir_ssim

        # Creating matrixs
        M_psnr = []
        M_uiqi = []
        M_image_fid = []
        M_ssim = []
        # Building watermaked images. Generating results
        for index, image_name in enumerate(image_names):
            # Instance
            waterk = DualWatermarkingScheme(
                "N;fpr-y7", 150, 62, 0.58, 0.24, matrix, matrix
            )

            # Load cover images
            dir_image = os.path.join(
            dir_covers,
            image_name
            )
            cover_array = iio.imread(dir_image)
            # Creating the watermarked images
            watermarked_array = waterk.insert(cover_array, watermark_as_list)
            # Creating directory
            dir_save = os.path.join(
                base,
                "watermarked_images/CS_" + sub_ind[item] + "/"
            )
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
            dir_save = (
                "%s%s" %
                (dir_save, image_name)
            )

            # Save stego image
            iio.imwrite(dir_save, watermarked_array)

            # Experimental analysis
            M_psnr.append(metr.psnr(cover_array, watermarked_array))
            print(" ")
            print("Experimental analysis")
            print(" ")
            print("PSNR: ", M_psnr[index])
            print(" ")
            # Saving results
            np.savetxt(dir_psnr, M_psnr, fmt="%.4e")

            M_uiqi.append(metr.uiqi(cover_array, watermarked_array))
            print(" ")
            print("UIQI: ", M_uiqi[index])
            print(" ")
            # Saving results
            np.savetxt(dir_uiqi, M_uiqi, fmt="%.4e")

            M_image_fid.append(metr.image_fid(cover_array, watermarked_array))
            print(" ")
            print("Image FID: ", M_image_fid[index])
            print(" ")
            # Saving results
            np.savetxt(dir_image_fid, M_image_fid, fmt="%.4e")

            M_ssim.append(metr.ssim(cover_array, watermarked_array))
            print(" ")
            print("SSIM: ", M_ssim[index])
            print(" ")
            # Saving results
            np.savetxt(dir_ssim, M_ssim, fmt="%.4e")

            # instance
            noise = NoisesClass(
                    watermarked_array,
                    str(index),
                    image_name.split(".")[0],
                    "CS_" + sub_ind[item]
                )

            noise.gaussian_noise([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])

            noise.minimum_filter_noise(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                )

            noise.gaussian_laplace(
                    [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
                )

            noise.percentile_filter([-20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])

            noise.uniform_filter([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

            noise.fourier_ellipsoid_filter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

            noise.fourier_gaussian_filter(
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
                )

            noise.fourier_uniform_filter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

            noise.salt_pepper_noise(
                    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
                )

            noise.cropping_noise([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

            noise.median_filter_noise([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

            noise.scaling_noise(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            )

            noise.rotation_noise(
                [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]
            )
            noise.jpeg_noise([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

        # Saving results
        np.savetxt(dir_psnr, M_psnr, fmt="%.4e")
        np.savetxt(dir_uiqi, M_uiqi, fmt="%.4e")
        np.savetxt(dir_image_fid, M_image_fid, fmt="%.4e")
        np.savetxt(dir_ssim, M_ssim, fmt="%.4e")


if __name__ == '__main__':
    main()
