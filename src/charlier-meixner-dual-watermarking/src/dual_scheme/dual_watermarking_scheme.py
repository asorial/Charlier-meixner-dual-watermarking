from inheritance.dual_watermarking_inheritance import DualWatermarkingSchemeInher
from moments_matrix.moments_class import SeparableMoments
from helpers.blocks_class import BlocksImage
from helpers.blocks_class_3d import BlocksImageChs
from helpers import progress_bar
from helpers.utils import *
from copy import deepcopy
import imageio.v3 as iio
import numpy as np
import hashlib
import hmac
import cv2
import os


class DualWatermarkingScheme(DualWatermarkingSchemeInher, SeparableMoments):
    """
        A dual watermaking scheme based on separable Discrete--Sobolev type orthogonal moments

        Robust layer: Dither Modulation of one moment coefficient per 8x8 block
        (channel ``CH``), unchanged with respect to the previous version.

        Fragile layer (revised, Oct 2026): block-wise keyed authentication tag

            t_k = Trunc_L( HMAC-SHA256_kappa( k || B_k^0 ) ),

        where k is the block index (4 bytes, big-endian), B_k^0 is the RGB 8x8 block
        after robust embedding with the LSBs of the L carrier pixels (channel ``CH``)
        set to zero, and kappa = SHA-256(key). The L bits of t_k are written into those
        LSBs. Verification recomputes t_k on the same canonical block B_k^0, so embedding
        and verification hash exactly the same data. The block index makes the tag
        position-dependent (copy-move / collage attacks are detected).

        Previous version: the same 3 key-derived bits were written in every block
        (pixels (0,0)-(0,2) of channel 2), so only those 3 LSBs were authenticated.
    """

    CH = 2  # channel carrying the robust and fragile watermarks (blue, RGB order)

    def __init__(self, key, qs, coefficient_num, p, x0, l_matrix, r_matrix,
                 tag_bits=16):

        DualWatermarkingSchemeInher.__init__(self, key, qs, coefficient_num, p, x0)

        SeparableMoments.__init__(self, l_matrix, r_matrix)

        # Fragile layer parameters
        self.kappa = hashlib.sha256(key.encode('utf-8')).digest()
        self.tag_bits = tag_bits
        # Carriers: first L pixels in raster order (rows 0-1), far from the
        # pixel (7,6) that dominates the robust coefficient
        self.carriers = [(i // 8, i % 8) for i in range(self.tag_bits)]
        # Last authentication map (blocks_rows x blocks_cols, True = tampered)
        self.last_tamper_map = None

    def separable_moments_direct(self, block):
        return self.direct_moments(block)

    def separable_moments_inverse(self, block):
        return self.inverse_moments(block)

    # ------------------------------------------------------------------
    # Robust layer
    # ------------------------------------------------------------------
    def embed_robust_block(self, block8x8, rwatermark):
        vac = vzig_zag_scan(
            self.separable_moments_direct(block8x8)
        )
        vac[self.coefficient_num] = self.insertion_method(
            vac[self.coefficient_num], rwatermark
        )
        return self.separable_moments_inverse(mzig_zag_scan(vac))

    def extract_robust_block(self, block8x8):
        vac = vzig_zag_scan(
            self.separable_moments_direct(block8x8)
        )
        return self.extraction_method(vac[self.coefficient_num])

    # ------------------------------------------------------------------
    # Fragile layer
    # ------------------------------------------------------------------
    def canonical_block(self, block_rgb):
        block = np.array(block_rgb, dtype=np.uint8, copy=True)
        for (y, x) in self.carriers:
            block[y, x, self.CH] &= 0xFE
        return block

    def block_tag(self, num_block, block_rgb):
        msg = int(num_block).to_bytes(4, 'big') + \
            self.canonical_block(block_rgb).tobytes()
        digest = hmac.new(self.kappa, msg, hashlib.sha256).digest()
        return "".join(format(byte, '08b') for byte in digest)[:self.tag_bits]

    def read_tag(self, block_rgb):
        return "".join(
            str(int(block_rgb[y, x, self.CH]) & 1) for (y, x) in self.carriers
        )

    def write_tag(self, block_rgb, tag):
        for bit, (y, x) in zip(tag, self.carriers):
            block_rgb[y, x, self.CH] = (int(block_rgb[y, x, self.CH]) & 0xFE) | int(bit)
        return block_rgb

    def authenticate(self, watermarked_array):
        """Return the tamper map (True = block whose tag does not verify)."""
        blocks = BlocksImageChs(watermarked_array)
        rows, cols = blocks.blocks_in_rows, blocks.blocks_in_cols
        tamper_map = np.zeros((rows, cols), dtype=bool)
        for index in range(blocks.max_num_blocks_chs()):
            r0, r1, c0, c1 = blocks.get_coord_block_chs(index)
            block_rgb = watermarked_array[r0:r1, c0:c1, :]
            tamper_map[index // cols, index % cols] = (
                self.read_tag(block_rgb) != self.block_tag(index, block_rgb)
            )
        self.last_tamper_map = tamper_map
        return tamper_map

    # ------------------------------------------------------------------
    # Embedding / extraction
    # ------------------------------------------------------------------
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
        progress_bar.printProgressBar(
            0, inst_block8x8.max_num_blocks_chs(),
            prefix='Progress:', suffix='Complete', length=50
        )
        for index in range(inst_block8x8.max_num_blocks_chs()):
            # 1) Robust layer on channel CH
            if index < len(watermark):
                block8x8 = inst_block8x8.get_block_ch(index, self.CH)
                block8x8 = self.embed_robust_block(block8x8, watermark[index])
                inst_block8x8.set_block_ch(block8x8, index, self.CH)
            # 2) Fragile layer: tag computed on the final RGB block
            r0, r1, c0, c1 = inst_block8x8.get_coord_block_chs(index)
            block_rgb = watermarked_array[r0:r1, c0:c1, :]
            self.write_tag(block_rgb, self.block_tag(index, block_rgb))
            progress_bar.printProgressBar(
                index + 1, inst_block8x8.max_num_blocks_chs(),
                prefix='Progress:', suffix='Complete', length=50
            )

        return watermarked_array

    def extraction(self, watermarked_array, image_names, save_tampered=True):
        # Initial value
        watermark = ""
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
        progress_bar.printProgressBar(
            0, inst_block8x8.max_num_blocks_chs(), prefix='Progress:',
            suffix='Complete', length=50
        )
        for index, block8x8 in enumerate(inst_block8x8.block_range_ch(self.CH)):
            watermark += self.extract_robust_block(block8x8)
            progress_bar.printProgressBar(
                index + 1, inst_block8x8.max_num_blocks_chs(),
                prefix='Progress:', suffix='Complete', length=50
            )

        # Fragile verification: full map of tampered blocks (no early exit)
        tamper_map = self.authenticate(watermarked_array)
        modified_blocks = list(np.flatnonzero(tamper_map))

        if modified_blocks:
            print("This document may not be an original "
                  "(%d tampered blocks)" % len(modified_blocks))
            if save_tampered:
                for item in modified_blocks:
                    coord = inst_block8x8_copy.get_coord_block_chs(int(item))
                    cv2.rectangle(watermarked_array_copy, (coord[2], coord[0]),
                        (coord[3], coord[1]), (255, 0, 0), 1)
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
                iio.imwrite(dir_stego_tamp, watermarked_array_copy)
        else:
            print("This document is original")

        return watermark[:64 * 64]
