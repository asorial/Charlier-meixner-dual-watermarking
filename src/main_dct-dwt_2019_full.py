import numpy as np
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

def embed_watermark(cover_image, watermark, k=40, arnold_iterations=10):
    """Inserta una marca de agua en una imagen RGB según el método del artículo."""
    # 1. Separar la imagen RGB en componentes
    R, G, B = cv2.split(cover_image)
    color_components = [R, G, B]
    watermarked_components = []
    
    # 2. Preparar la marca de agua
    # a. Transformada de Arnold
    scrambled_watermark = arnold_transform(watermark, arnold_iterations)
    
    # b. Convertir a binario (9216x8)
    watermark_bin = np.unpackbits(scrambled_watermark).reshape(-1, 8)
    
    # c. Dividir en 3 partes y rellenar
    parts = np.split(watermark_bin, 3)
    watermark_blocks = []
    for part in parts:
        block = np.pad(part.flatten(), (0, 25600 - part.size), 'constant')
        watermark_blocks.append(block.reshape(160, 160))
    
    # Procesar cada componente de color
    for c_idx, I_c in enumerate(color_components):
        # 3. Aplicar DCT global
        I_c_DCT = cv2.dct(I_c.astype(np.float64))
        
        # 4. Aplicar DWT (1 nivel, Haar)
        LL, (LH, HL, HH) = pywt.dwt2(I_c_DCT, 'haar')
        bands = [LL, LH, HL, HH]
        
        # 5. Procesar bloque de marca de agua
        W_c = watermark_blocks[c_idx]
        W_c_DCT = cv2.dct(W_c.astype(np.float64))
        
        # 6. Dividir en sub-bloques 80x80
        sub_blocks = [
            W_c_DCT[:80, :80],
            W_c_DCT[:80, 80:],
            W_c_DCT[80:, :80],
            W_c_DCT[80:, 80:]
        ]
        
        # 7. Insertar en bandas DWT
        watermarked_bands = []
        for band, sub_block in zip(bands, sub_blocks):
            # Crear matriz 512x512 con el sub-bloque
            embedded_block = np.zeros_like(band)
            embedded_block[:80, :80] = sub_block
            # Incrustar con factor de escala
            watermarked_band = band + embedded_block * k
            watermarked_bands.append(watermarked_band)
        
        # 8. Reconstrucción
        LL_w, LH_w, HL_w, HH_w = watermarked_bands
        I_c_DCT_w = pywt.idwt2((LL_w, (LH_w, HL_w, HH_w)), 'haar')
        I_c_w = cv2.idct(I_c_DCT_w)
        watermarked_components.append(I_c_w)
    
    # 9. Combinar componentes
    watermarked_image = cv2.merge(watermarked_components)
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark(original_image, watermarked_image, k=40, arnold_iterations=10, watermark_size=96):
    """Extrae la marca de agua de una imagen según el método del artículo."""
    # 1. Separar componentes
    R_o, G_o, B_o = cv2.split(original_image)
    R_w, G_w, B_w = cv2.split(watermarked_image)
    orig_components = [R_o, G_o, B_o]
    wm_components = [R_w, G_w, B_w]
    extracted_parts = []
    
    for orig, wm in zip(orig_components, wm_components):
        # 2. Aplicar DCT + DWT a ambas imágenes
        orig_DCT = cv2.dct(orig.astype(np.float64))
        wm_DCT = cv2.dct(wm.astype(np.float64))
        
        orig_LL, _ = pywt.dwt2(orig_DCT, 'haar')
        wm_LL, wm_coeffs = pywt.dwt2(wm_DCT, 'haar')
        wm_LH, wm_HL, wm_HH = wm_coeffs
        
        # 3. Extraer marcas de todas las bandas
        extracted_blocks = []
        for band in [wm_LL, wm_LH, wm_HL, wm_HH]:
            # Extraer componente
            orig_band, _ = pywt.dwt2(orig_DCT, 'haar')
            W = (band - orig_band) / k
            extracted_blocks.append(W[:80, :80])
        
        # 4. Reconstruir bloque DCT
        W_DCT = np.vstack((
            np.hstack((extracted_blocks[0], extracted_blocks[1])),
            np.hstack((extracted_blocks[2], extracted_blocks[3]))
        ))
        
        # 5. Aplicar IDCT
        W_block = cv2.idct(W_DCT)
        extracted_parts.append(W_block[:3072].flatten())
    
    # 6. Combinar y descrambling
    combined_bits = np.concatenate(extracted_parts)
    binary_watermark = np.packbits(combined_bits[:watermark_size*watermark_size*8])
    extracted_watermark = binary_watermark[:watermark_size*watermark_size].reshape(watermark_size, watermark_size)
    return inverse_arnold(extracted_watermark, arnold_iterations)




# Cargar imágenes
cover_image = cv2.imread("cover_image.jpg")  # 1024x1024 RGB
watermark = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)  # 96x96 escala de grises

# Incrustar marca de agua
watermarked_image = embed_watermark(
    cover_image,
    watermark,
    k=40,               # Factor de escala
    arnold_iterations=10
)

# Extraer marca de agua
extracted_watermark = extract_watermark(
    cover_image,
    watermarked_image,
    k=40,
    arnold_iterations=10,
    watermark_size=96
)

# Guardar resultados
cv2.imwrite("watermarked.jpg", watermarked_image)
cv2.imwrite("extracted_watermark.png", extracted_watermark)