
from math import log10, log
import numpy as np
import cv2

class Metrics:
    def __init__(self):
        self.C1 = (0.01 * 255) ** 2
        self.C2 = (0.03 * 255) ** 2
        self.C3 = self.C2 / 2

    def mse(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        diff_pow = (np.abs(C) - np.abs(S)) ** 2
        if len(dims_cover) == 2:
            return sum(sum(diff_pow)) / (m * n)
        elif len(dims_cover) == 3:
            return sum(sum(sum(diff_pow))) / (m * n * dims_cover[2])

    def psnr(self, cover_array, stego_array):
        RMSE = self.mse(cover_array, stego_array)
        return 10 * log10(255 ** 2 / RMSE) if RMSE else 100

    def uiqi(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        if len(cover_array.shape) == 2:
            C_m = sum(sum(C)) / (m * n)
            S_m = sum(sum(S)) / (m * n)
            diff_pow = (C - C_m) ** 2
            sigma2c = sum(sum(diff_pow)) / (m * n - 1)
            diff_pow = (S - S_m) ** 2
            sigma2s = sum(sum(diff_pow)) / (m * n - 1)
            prod_diff = (C - C_m) * (S - S_m)
            sigmacs = sum(sum(prod_diff)) / (m * n - 1)
        elif len(cover_array.shape) == 3:
            C_m = sum(sum(sum(C))) / (m * n * dims_cover[2])
            S_m = sum(sum(sum(S))) / (m * n * dims_cover[2])
            diff_pow = (C - C_m) ** 2
            sigma2c = sum(sum(sum(diff_pow))) / (m * n * dims_cover[2] - 1)
            diff_pow = (S - S_m) ** 2
            sigma2s = sum(sum(sum(diff_pow))) / (m * n * dims_cover[2] - 1)
            prod_diff = (C - C_m) * (S - S_m)
            sigmacs = sum(sum(sum(prod_diff))) / (m * n * dims_cover[2] - 1)
        aux = 4 * sigmacs * C_m * S_m
        return aux / ((sigma2c + sigma2s) * (C_m ** 2 + S_m ** 2))

    def image_fid(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        if len(cover_array.shape) == 2:
            return (
                1 - sum(sum((C - S) ** 2)) / sum(sum(C ** 2))
                )
        elif len(cover_array.shape) == 3:
            return (
                1 - sum(sum(sum((C - S) ** 2))) / sum(sum(sum(C ** 2)))
                )

    def correlation_quality(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        if len(cover_array.shape) == 2:
            return 1 - sum(sum(np.sqrt(C * S))) / sum(sum(C))
        elif len(cover_array.shape) == 3:
            return 1 - sum(sum(sum(np.sqrt(C * S)))) / sum(sum(sum(C)))

    def structure_content(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        if len(cover_array.shape) == 2:
            return sum(sum(C ** 2)) / sum(sum(S ** 2))
        elif len(cover_array.shape) == 3:
            return sum(sum(sum(C ** 2))) / sum(sum(sum(S ** 2)))

    def absolute_difference(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        C, S = cover_array * 1.0, stego_array * 1.0
        if len(cover_array.shape) == 2:
            return sum(sum(np.abs(C - S))) / (m * n)
        elif len(cover_array.shape) == 3:
            return sum(sum(sum(np.abs(C - S)))) / (3 * m * n)

    def hist_sim(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        if len(cover_array.shape) == 2:
            histc_1 = cv2.calcHist([cover_array], [0], None, [256], [0, 256])
            hists_1 = cv2.calcHist([stego_array], [0], None, [256], [0, 256])
            hs = 0
            for i in range(256):
                diff = (histc_1[i][0] - hists_1[i][0]) / (m * n)
                hs += abs(diff)
        elif len(cover_array.shape) == 3:
            histc_1 = cv2.calcHist([cover_array], [0], None, [256], [0, 256])
            histc_2 = cv2.calcHist([cover_array], [1], None, [256], [0, 256])
            histc_3 = cv2.calcHist([cover_array], [2], None, [256], [0, 256])
            hists_1 = cv2.calcHist([stego_array], [0], None, [256], [0, 256])
            hists_2 = cv2.calcHist([stego_array], [1], None, [256], [0, 256])
            hists_3 = cv2.calcHist([stego_array], [2], None, [256], [0, 256])
            hs = 0
            for i in range(256):
                aux_1 = histc_1[i][0] + histc_2[i][0] + histc_3[i][0]
                aux_2 = hists_1[i][0] + hists_2[i][0] + hists_3[i][0]
                diff = (aux_1 - aux_2) / (m * n * dims_cover[2])
                UIQIhs += abs(diff)
        return hs

    def rel_entropy(self, cover_array, stego_array):
        dims_cover = cover_array.shape
        m = dims_cover[0]
        n = dims_cover[1]
        if len(cover_array.shape) == 2:
            histc_1 = cv2.calcHist([cover_array], [0], None, [256], [0, 256])
            hists_1 = cv2.calcHist([stego_array], [0], None, [256], [0, 256])
            rel_ent = 0
            for i in range(256):
                prob_cover = histc_1[i][0] / sum(histc_1)[0]
                prob_stego = hists_1[i][0] / sum(histc_1)[0]
        elif len(cover_array.shape) == 3:
            histc_1 = cv2.calcHist([cover_array], [0], None, [256], [0, 256])
            histc_2 = cv2.calcHist([cover_array], [1], None, [256], [0, 256])
            histc_3 = cv2.calcHist([cover_array], [2], None, [256], [0, 256])
            hists_1 = cv2.calcHist([stego_array], [0], None, [256], [0, 256])
            hists_2 = cv2.calcHist([stego_array], [1], None, [256], [0, 256])
            hists_3 = cv2.calcHist([stego_array], [2], None, [256], [0, 256])
            rel_ent = 0
            for i in range(256):
                aux_1 = histc_1[i][0] + histc_2[i][0] + histc_3[i][0]
                aux_2 = hists_1[i][0] + hists_2[i][0] + hists_3[i][0]
                sum_all = sum(histc_1)[0] + sum(histc_2)[0] + sum(histc_3)[0]
                prob_cover = aux_1 / sum_all
                prob_stego = aux_2 / sum_all
        if prob_cover != 0 and prob_stego != 0:
            rel_ent += prob_cover * abs(log(prob_cover / prob_stego))
        return rel_ent

    def ber(self, binary_seq_1, binary_seq_2):
        s = 0
        n = len(binary_seq_1)
        for i in range(n):
            if binary_seq_1[i] != binary_seq_2[i]:
                s += 1
        return float(s)/len(binary_seq_1)

    def ssim(self, cover_array, stego_array):
        if len(cover_array.shape) not in [2, 3]:
            raise ValueError("Unsupported array dimensions for SSIM calculation. Only 2D and 3D arrays are supported.")

        C, S = cover_array * 1.0, stego_array * 1.0  # Convertir a float

        # Convertir imágenes 2D a 3D con un canal
        if len(cover_array.shape) == 2:
            C = C.reshape(cover_array.shape[0], cover_array.shape[1], 1)
            S = S.reshape(stego_array.shape[0], stego_array.shape[1], 1)

        m, n, ch = C.shape
        mu_c = np.mean(C, axis=(0, 1))  # Media por canal
        mu_s = np.mean(S, axis=(0, 1))
        sigma_c = np.var(C, axis=(0, 1), ddof=1)  # Varianza (ddof=1 para consistencia)
        sigma_s = np.var(S, axis=(0, 1), ddof=1)

        # Aplanar imágenes y calcular covarianza por canal
        C_flat = C.reshape(-1, ch)
        S_flat = S.reshape(-1, ch)
        centered_c = C_flat - mu_c
        centered_s = S_flat - mu_s
        sigma_cs = np.sum(centered_c * centered_s, axis=0) / (m * n - 1)  # Covarianza

        # Cálculo SSIM por canal
        num = (2 * mu_c * mu_s + self.C1) * (2 * sigma_cs + self.C2)
        den = (mu_c**2 + mu_s**2 + self.C1) * (sigma_c + sigma_s + self.C2)
        ssim_per_channel = num / den

        return np.mean(ssim_per_channel)  # Promedio sobre canales

    def ncc(self, watermark_array, extracted_watermark_array):
        """
        Calcula el NCC según el artículo:
        "Robust image watermarking scheme with transform domain technique..."

        Parámetros:
        watermark_array -- Marca de agua original (2D/3D)
        extracted_watermark_array -- Marca de agua extraída (mismas dimensiones)

        Retorna:
        Valor NCC en [-1, 1] donde 1 indica máxima similitud
        """
        # Convertir a vectores 1D de precisión doble
        orig = watermark_array.ravel().astype(np.float64)
        extr = extracted_watermark_array.ravel().astype(np.float64)

        # Normalizar si los valores están en [0, 255]
        if orig.max() > 1.0 or extr.max() > 1.0:
            orig /= 255.0
            extr /= 255.0

        # Calcular producto punto y normas
        dot_product = np.dot(orig, extr)
        norm_orig = np.linalg.norm(orig)
        norm_extr = np.linalg.norm(extr)

        # Manejar casos especiales (normas cercanas a cero)
        if norm_orig < 1e-12 and norm_extr < 1e-12:
            return 1.0  # Ambas son cero -> idénticas
        elif norm_orig < 1e-12 or norm_extr < 1e-12:
            return 0.0  # Solo una es cero -> no hay correlación

        # Calcular NCC y asegurar valor dentro de [-1, 1]
        ncc_val = dot_product / (norm_orig * norm_extr)
        return np.clip(ncc_val, -1.0, 1.0)

    def calculate_ber_ncc(self, original_wm, extracted_wm):
        # Normalizar imágenes a uint8 [0, 255] si no lo son
        if original_wm.dtype != np.uint8:
            original_wm = cv2.normalize(original_wm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if extracted_wm.dtype != np.uint8:
            extracted_wm = cv2.normalize(extracted_wm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Binarización con Otsu (maneja múltiples canales)
        def binarize_image(img):
            if len(img.shape) > 2:  # Imagen a color
                bin_img = np.zeros_like(img)
                for i in range(img.shape[2]):
                    _, channel_bin = cv2.threshold(img[..., i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    bin_img[..., i] = channel_bin
                return bin_img
            else:  # Imagen en escala de grises
                _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return bin_img
        # Binarizar ambas marcas de agua
        bin_orig = binarize_image(original_wm)
        bin_ext = binarize_image(extracted_wm)

        # Calcular BER (porcentaje de píxeles diferentes)
        ber = np.sum(bin_orig != bin_ext) / bin_orig.size

        # Calcular NCC (manejo de casos con varianza cero)
        orig_flat = original_wm.astype(np.float64).flatten()
        ext_flat = extracted_wm.astype(np.float64).flatten()

        # Caso especial: ambas imágenes son constantes
        if np.all(orig_flat == orig_flat[0]) and np.all(ext_flat == ext_flat[0]):
            ncc = 1.0 if orig_flat[0] == ext_flat[0] else 0.0
        else:
            ncc_matrix = np.corrcoef(orig_flat, ext_flat)
            ncc = ncc_matrix[0, 1]
            if np.isnan(ncc):  # Manejar correlación indefinida
                ncc = 0.0

        return ber, ncc