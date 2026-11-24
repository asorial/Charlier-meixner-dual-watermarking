
import numpy as np


class DAT:  # Discret Arnold Transforms

    def dat2(self, input_array):
        s = input_array.shape
        output_array = np.array(input_array)
        for x in range(s[1]):
            for y in range(s[0]):
                output_array[
                        (x + y) % s[1], (2 * x + y) % s[0]
                    ] = input_array[x, y]
        return output_array