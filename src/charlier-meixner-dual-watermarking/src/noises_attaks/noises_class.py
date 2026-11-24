

from attacks.noises import gaussian_noise, salt_pepper_noise
from helpers.utils import cropping_noise, random_list
from helpers import progress_bar, blocks_class
from scipy import ndimage
import imageio.v3 as iio
import numpy as np
import imageio
import cv2
import os


class NoisesClass():
    def __init__(self, watermarked_array, folder_name, image_name, method_name):
        self.watermarked_array = watermarked_array
        self.folder_name = folder_name
        self.image_name = image_name

        # Directory
        base = "src/static"
        base = os.path.join(os.getcwd(), base)
        self.noises_dirs = os.path.join(
                base,
                "watermarked_with_nosises",
                method_name,
            )
        if not os.path.exists(self.noises_dirs):
            os.makedirs(self.noises_dirs)

    def gaussian_noise(self, percent_noise_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying gaussian noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(percent_noise_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, percent_index in enumerate(percent_noise_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = gaussian_noise(image_array, percent_noise=percent_index)
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "0_gaussian",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_gaussian_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            iio.imwrite(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(percent_noise_list),
                prefix='Progress:', suffix='Complete', length=50)

    def minimum_filter_noise(self, size_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying minimum filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(size_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, size_index in enumerate(size_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.minimum_filter(
                image_array,
                size=size_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "1_minimum_filter",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_minimum_filter_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            iio.imwrite(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(size_list),
                prefix='Progress:', suffix='Complete', length=50)

    def gaussian_laplace(self, sigma_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying gaussian laplace noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        progress_bar.printProgressBar(0, len(sigma_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, sigma_index in enumerate(sigma_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.gaussian_laplace(
                image_array,
                sigma=sigma_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "2_gaussian_laplace",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_gaussian_laplace_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            iio.imwrite(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(sigma_list),
                prefix='Progress:', suffix='Complete', length=50)

    def percentile_filter(self, percentile_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying percentile filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # The percentile parameter may be less then zero,
        # i.e., percentile = -20 equals percentile = 80

        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(percentile_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, percentile_index in enumerate(percentile_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.percentile_filter(
                image_array,
                percentile=percentile_index,
                size=1
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "3_percentile_filter",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_percentile_filter_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            iio.imwrite(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(percentile_list),
                prefix='Progress:', suffix='Complete', length=50)

    def uniform_filter(self, size_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying uniform filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # The sizes of the uniform filter are given for each axis
        #  as a sequence, or as a single number, in which case the
        # size is equal for all axes.

        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(size_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, size_index in enumerate(size_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.uniform_filter(
                image_array,
                size=size_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "4_uniform_filter",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_uniform_filter_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            iio.imwrite(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(size_list),
                prefix='Progress:', suffix='Complete', length=50)

    def fourier_ellipsoid_filter(self, size_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying fourier ellipsoid filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # The size of the box used for filtering. If a float, size is the
        # same for all axes. If a sequence, size has to contain one value
        #  for each axis.

        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(size_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, size_index in enumerate(size_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.fourier_ellipsoid(
                image_array,
                size=size_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "5_fourier_ellipsoid_filter",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_fourier_ellipsoid_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            iio.imwrite(save_dir, image_array.astype(np.uint8))
            progress_bar.printProgressBar(index+1, len(size_list),
                prefix='Progress:', suffix='Complete', length=50)

    def fourier_gaussian_filter(self, sigma_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying fourier gaussian filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        progress_bar.printProgressBar(0, len(sigma_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, sigma_index in enumerate(sigma_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.fourier_gaussian(image_array, sigma=sigma_index)
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "6_fourier_gaussian",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_fourier_gaussian.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            imageio.imsave(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(sigma_list),
                prefix='Progress:', suffix='Complete', length=50)

    def fourier_uniform_filter(self, size_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying fourier uniform filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(size_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, size_index in enumerate(size_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.fourier_uniform(
                image_array,
                size=size_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "7_fourier_uniform_filter",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_fourier_uniform_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            imageio.imsave(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(size_list),
                prefix='Progress:', suffix='Complete', length=50)

    def salt_pepper_noise(self, density_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying salt and pepper noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(density_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, density_index in enumerate(density_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = salt_pepper_noise(
                image_array,
                density=density_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "8_salt_pepper_noise",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_salt_pepper_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            imageio.imsave(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(density_list),
                prefix='Progress:', suffix='Complete', length=50)

    def cropping_noise(self, percent_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying cropping noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(percent_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, percent in enumerate(percent_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            blocks_instance = blocks_class.BlocksImage(image_array, 32, 32)
            poslist = random_list(
                0.57, 0.24,
                list(range(blocks_instance.max_num_blocks()))
            )[:int(blocks_instance.max_num_blocks() * percent / 100)]
            for pos in poslist:
                blocks_instance.set_block(cropping_noise(
                    blocks_instance.get_block(pos), 32, 32), pos
                )

            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "9_cropping_noise",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_cropping_noise.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            imageio.imsave(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(percent_list),
                prefix='Progress:', suffix='Complete', length=50)

    def median_filter_noise(self, size_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying median filter noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(size_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, size_index in enumerate(size_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Applying noise to the image
            image_array = ndimage.median_filter(
                image_array,
                size=size_index
            )
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "10_median_filter",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_median_filter.%s"
                %
                (save_dir, "bmp")
            )
            # Saving watermarked image with noise
            imageio.imsave(save_dir, image_array)
            progress_bar.printProgressBar(index+1, len(size_list),
                prefix='Progress:', suffix='Complete', length=50)

    def scaling_noise(self, scale_factor_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying scaling noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(scale_factor_list), prefix='Progress:',
                                    suffix='Complete', length=50)

        # Get original dimensions
        original_height, original_width = self.watermarked_array.shape[:2]

        for index, scale_factor in enumerate(scale_factor_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)

            # Calculate new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Ensure minimum dimensions of 1 pixel
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            # Resize the image
            resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Create canvas with original size
            if scale_factor > 1.0:  # Zoom in - crop center
                start_x = (new_width - original_width) // 2
                start_y = (new_height - original_height) // 2
                result = resized[
                    start_y:start_y + original_height,
                    start_x:start_x + original_width
                ]
            else:  # Zoom out - place on black background
                result = np.zeros_like(image_array)
                start_x = (original_width - new_width) // 2
                start_y = (original_height - new_height) // 2
                result[
                    start_y:start_y + new_height,
                    start_x:start_x + new_width
                ] = resized

            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "11_scaling_noise",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(
                save_dir,
                f"{self.image_name}_{index}_scale_{scale_factor}.bmp"
            )

            # Saving scaled image
            imageio.imsave(save_path, result)
            progress_bar.printProgressBar(index + 1, len(scale_factor_list),
                                        prefix='Progress:', suffix='Complete', length=50)

    def rotation_noise(self, angle_list, background=(0, 0, 0)):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying rotation noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        progress_bar.printProgressBar(0, len(angle_list), prefix='Progress:',
            suffix='Complete', length=50)

        h, w = self.watermarked_array.shape[:2]
        center = (w // 2, h // 2)

        for index, angle in enumerate(angle_list):
            img_copy = np.copy(self.watermarked_array)

            # Handle different color channels
            if len(img_copy.shape) == 2:  # Grayscale
                bg_value = background[0] if isinstance(background, tuple) else background
            else:  # Color image
                bg_value = background

            # Create and apply rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img_copy,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=bg_value
            )

            # Create save directory
            save_dir = os.path.join(self.noises_dirs, "12_rotation_noise", self.folder_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.image_name}_{index}_angle_{angle}.bmp")

            # Save image
            imageio.imsave(save_path, rotated)
            progress_bar.printProgressBar(index + 1, len(angle_list),
                prefix='Progress:', suffix='Complete', length=50)

    def jpeg_noise(self, quality_list):
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Applying jpeg noise:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(0, len(quality_list), prefix='Progress:',
            suffix='Complete', length=50)
        for index, qual in enumerate(quality_list):
            # Creating a copy of the watermarked image
            image_array = np.copy(self.watermarked_array)
            # Creating directory
            save_dir = os.path.join(
                self.noises_dirs,
                "13_jpeg_noise",
                self.folder_name
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(
                save_dir,
                self.image_name + "_" + str(index),
            )
            save_dir = (
                "%s_jpeg_noise.%s"
                %
                (save_dir, "jpg")
            )
            # Saving watermarked image with noise
            imageio.imsave(save_dir, image_array, quality=qual)
            progress_bar.printProgressBar(index+1, len(quality_list),
                prefix='Progress:', suffix='Complete', length=50)