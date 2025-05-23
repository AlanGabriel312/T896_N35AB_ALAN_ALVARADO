import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Carrega as imagens
# foreground: pessoas sobre fundo verde
# background: nova cena
foreground = cv2.imread('img_fundo_verde_3.png')
background = cv2.imread('background_1.png')

# 2) Redimensiona o background para ter o mesmo tamanho do foreground
h, w = foreground.shape[:2]
background = cv2.resize(background, (w, h))

# 3) Converte foreground de BGR para HSV (mais fácil isolar cores)
hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

# 4) Define intervalo de verde (pode precisar ajustar)
verde_menor = np.array([35, 50, 50])   # matiz≈35°, saturação e valor mínimos
verde_maior = np.array([85, 255, 255]) # matiz≈85°, saturação e valor máximos

# 5) Cria máscara do verde
mascara_verde = cv2.inRange(hsv, verde_menor, verde_maior)

# 6) Inverte a máscara para “pegar” a pessoa
mascara_pessoa = cv2.bitwise_not(mascara_verde)

# 7) Extrai pessoa (foreground sem fundo verde)
pessoa = cv2.bitwise_and(foreground, foreground, mask=mascara_pessoa)

# 8) Extrai região correspondente do background onde era verde
regiao_bg = cv2.bitwise_and(background, background, mask=mascara_verde)

# 9) Soma as duas partes para obter o resultado final
chroma_resultado = cv2.add(pessoa, regiao_bg)

# 10) Converte de BGR para RGB para exibição com Matplotlib
chroma_resultado_rgb = cv2.cvtColor(chroma_resultado, cv2.COLOR_BGR2RGB)
foreground  = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
background  = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# 11) Mostra o resultado
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(foreground)
plt.title('Foreground')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(background)
plt.title('Background')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(chroma_resultado_rgb)
plt.title('Resultado Chroma Key')
plt.axis('off')

plt.tight_layout()
plt.show()
