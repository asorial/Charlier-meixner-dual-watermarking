

class BlocksImageChs():
    def __init__(self, image_array, sblock_rows=8, sblock_cols=8):
        self.image_array = image_array
        self.size_block_rows = sblock_rows
        self.size_block_cols = sblock_cols
        self.blocks_in_rows = (
            self.image_array.shape[0] // self.size_block_rows
        )
        self.blocks_in_cols = (
            self.image_array.shape[1] // self.size_block_cols
        )

    def get_image(self):
        return self.image_array

    def get_ch(self, chan):
        return self.image_array[:, :, chan]

    def max_num_blocks_chs(self):
        return self.blocks_in_rows * self.blocks_in_cols

    def image_size(self):
        return self.image_array.shape

    def get_coord_block_chs(self, num_block):
        if num_block < self.max_num_blocks_chs():
            L = []
            row_block = int(num_block / self.blocks_in_cols)
            col_block = num_block % self.blocks_in_cols
            L.append(row_block * self.size_block_rows)
            L.append((row_block + 1) * self.size_block_rows)
            L.append(col_block * self.size_block_cols)
            L.append((col_block + 1) * self.size_block_cols)
            return L
        raise Exception("There is no such block")

    def get_block_ch(self, num_block, chan):
        try:
            pos = self.get_coord_block_chs(num_block)
            matrix = (
                self.get_ch(chan)[
                    pos[0]:pos[1],
                    pos[2]:pos[3]
                ]
            )
            if matrix.shape != (1, 1):
                return matrix
            else:
                return matrix[0][0]
        except Exception:
            return None

    def set_block_ch(self, block, num_block, chan):
        pos = self.get_coord_block_chs(num_block)
        self.get_ch(chan)[pos[0]:pos[1], pos[2]:pos[3]] = block

    def block_range_ch(self, chan):
        for i in range(self.max_num_blocks_chs()):
            yield self.get_block_ch(i, chan)
