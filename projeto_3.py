import cv2
import numpy as np
import matplotlib.pyplot as plt

# Projeto 3: Detecção de Folhas Saudáveis e Danificadas
# Segmenta áreas saudáveis (verde) e danificadas (amareladas/marrom) usando HSV e morfologia.

# 1) Carrega a imagem da folha
imagem = cv2.imread('img_folha_7.JPG')
if imagem is None:
    raise FileNotFoundError("Não foi possível carregar 'img_folha_1.JPG'.")

# 2) Converte para HSV (tono, saturação, valor)
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# 3) Define limites de cor em HSV
#   - Saudável: tons de verde
lim_saud_baixo = np.array([36,  25, 25])   # H≈36°, S/V mínimos
lim_saud_alto = np.array([86, 255,255])   # H≈86°
#    - Danificada AMARELADA/MARROM
lim_dan1_baixo = np.array([10,  50, 50])   # H≈10°
lim_dan1_alto  = np.array([35, 255,255])   # H≈35°
#    - Danificada PRETA (qualquer matiz/saturação, valor ≤ 50)
lim_dan2_baixo = np.array([  0,   0,  0])
lim_dan2_alto  = np.array([180, 255, 50])

# 4) Cria máscaras binárias
mascara_saudavel = cv2.inRange(hsv, lim_saud_baixo, lim_saud_alto)
mask_dan1 = cv2.inRange(hsv, lim_dan1_baixo, lim_dan1_alto)
mask_dan2 = cv2.inRange(hsv, lim_dan2_baixo, lim_dan2_alto)
mascara_danificado = cv2.bitwise_or(mask_dan1, mask_dan2)

# 5) Operações morfológicas para limpar ruídos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mascara_saudavel = cv2.morphologyEx(mascara_saudavel, cv2.MORPH_CLOSE, kernel)
mascara_saudavel = cv2.morphologyEx(mascara_saudavel, cv2.MORPH_OPEN,  kernel)
mascara_danificado  = cv2.morphologyEx(mascara_danificado,  cv2.MORPH_CLOSE, kernel)
mascara_danificado  = cv2.morphologyEx(mascara_danificado,  cv2.MORPH_OPEN,  kernel)

# 6) Extrai regiões segmentadas usando as máscaras
folha_saudavel = cv2.bitwise_and(imagem, imagem, mask=mascara_saudavel) 
folha_danificada  = cv2.bitwise_and(imagem, imagem, mask=mascara_danificado)

# 7) trasnfoma de BGR para RGB

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

folha_saudavel = cv2.cvtColor(folha_saudavel, cv2.COLOR_BGR2RGB)

folha_danificada = cv2.cvtColor(folha_danificada, cv2.COLOR_BGR2RGB)


# 8) Exibe resultados lado a lado
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagem)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(folha_saudavel)
plt.title('Áreas Saudáveis')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(folha_danificada)
plt.title('Áreas Danificadas')
plt.axis('off')

plt.tight_layout()
plt.show()

# 9) Sobreposição das regiões no original
overlay = imagem.copy()
overlay[mascara_saudavel > 0] = (0, 255, 0)  # destaca verde nas saudáveis
overlay[mascara_danificado  > 0] = (0, 0, 255)  # destaca vermelho nas danificadas

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Overlay: Verde=Saudáveis  Vermelho=Danificadas')
plt.axis('off')
plt.show()

