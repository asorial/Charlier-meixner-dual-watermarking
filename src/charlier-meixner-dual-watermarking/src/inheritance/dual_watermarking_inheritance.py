
from helpers.blocks_class_3d import BlocksImageChs
from quantization.scalar import UniformQuantizer
from helpers import progress_bar
from embedding.qim import dm
from helpers.utils import *
from copy import deepcopy
import imageio.v3 as iio
import cv2
import os


class DualWatermarkingSchemeInher():
    """
        Dual Watermarking Scheme Inher
    """

    def __init__(self, key, qs, coefficient_num, p, x0):
        self.qs = qs
        self.q = UniformQuantizer(step=self.qs)
        self.d = dm.BinaryDither(step=self.qs, d0=-self.qs / 4)
        self.emb = dm.BinaryDM(self.q, self.d)
        self.coefficient_num = coefficient_num
        self.random_items = random_list(x0, p, list(range(256)))[:3]
        self.fragile_watermark = "".join(
            sha256_to_key_bin(key)[item] for item in self.random_items
        )

    def separable_moments_direct(self, block):
        pass

    def separable_moments_inverse(self, block):
        pass

    def insertion_method(self, amplitude, bit):
        return self.emb.embed(amplitude, bit)

    def zig_zag_scan(self, block):
        return vzig_zag_scan(block)

    def rev_zig_zag_scan(self, coefficients):
        return mzig_zag_scan(coefficients)

    def quantized(self, block):
        # Applying the direct moments
        coefficients_block = self.separable_moments_direct(block)
        # Quantifying
        return quant(coefficients_block, 77)

    def unquantized(self, block):
        # Multiply the previous matrix by the quantification matrix
        coefficients_block = quantinv(block, 77)
        # Apply the inverse moment transform
        return self.separable_moments_inverse(coefficients_block)

    def insertion_method_lsb(self, pixels_list, bin_msg):
        pixels = []
        for index, elem in enumerate(bin_msg):
            pixels.append(replace(pixels_list[index], bin_msg[index]))
        return pixels

    def modify_block_ins_dual(self, block, bin):
        pass

    def modify_block_ins_fragile(self, block):
        pass

    def modify_block_ext(self, block):
        pass

    def insert(self, cover_array, watermark):
        # Creating copy
        watermarked_array = deepcopy(cover_array)
        # Instance
        inst_block8x8 = BlocksImageChs(watermarked_array)
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Embedding the bits of the watermark:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(
            0, inst_block8x8.max_num_blocks_chs(),
            prefix='Progress:',
            suffix='Complete', length=50
        )
        for index, block8x8 in enumerate(inst_block8x8.block_range_ch(2)):
            # Modifying block
            if index < len(watermark):
                block8x8 = self.modify_block_ins_dual(block8x8, watermark[index])
            else:
                block8x8 = self.modify_block_ins_fragile(block8x8)
            inst_block8x8.set_block_ch(block8x8, index, 2)
            progress_bar.printProgressBar(
                index+1, inst_block8x8.max_num_blocks_chs(),
                prefix='Progress:', suffix='Complete', length=50
            )

        return watermarked_array

    def extraction_method(self, amplitude):
        return str(self.emb.extract(amplitude))

    def extraction_method_lsb(self, pixels_list):
        bin_msg = ""
        for elem in pixels_list:
            bin_msg += ext_lsb(elem)
        return bin_msg

    def extraction(self, watermarked_array, image_names):
        # Initial value
        watermark = ""
        modified_blocks = []
        # Watermarked array copy
        watermarked_array_copy = deepcopy(watermarked_array)
        # Instance
        inst_block8x8 = BlocksImageChs(watermarked_array)
        inst_block8x8_copy = BlocksImageChs(watermarked_array_copy)
        print("-------------------------------------------------------------" +
            "-----------------")
        print("Extracting the bits corresponding to the watermark:")
        print("-------------------------------------------------------------" +
            "-----------------")
        # Initial call to print 0% progress
        progress_bar.printProgressBar(
            0, inst_block8x8.max_num_blocks_chs(), prefix='Progress:',
            suffix='Complete', length=50
        )
        for index, block8x8 in enumerate(inst_block8x8.block_range_ch(2)):
            # Extracting bits
            fragile_watermark_ext, rwatermark = self.modify_block_ext(block8x8)
            watermark += rwatermark
            if fragile_watermark_ext != self.fragile_watermark:
                modified_blocks.append(index)
            progress_bar.printProgressBar(
                index+1, inst_block8x8.max_num_blocks_chs(),
                prefix='Progress:', suffix='Complete', length=50
            )

        if modified_blocks:
            print("This document may not be an original")
            for item in modified_blocks:
                coord = inst_block8x8_copy.get_coord_block_chs(item)
                cv2.rectangle(watermarked_array_copy, (coord[2], coord[0]),
                (coord[3], coord[1]), (255, 0, 0), 1)
            # Save image with noise
            dir_stego_tamp = os.path.join(
                "src/static",
                "tampered_images",
            )
            if not os.path.exists(dir_stego_tamp):
                os.makedirs(dir_stego_tamp)
            dir_stego_tamp = os.path.join(
                dir_stego_tamp,
                "tampered_%s" % image_names
            )
            # Saving tampered image
            iio.imwrite(dir_stego_tamp, watermarked_array_copy)
        else:
            print("This document is original")

        return watermark[:64 * 64]
