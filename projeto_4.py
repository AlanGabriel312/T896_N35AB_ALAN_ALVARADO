import cv2
import numpy as np
import matplotlib.pyplot as plt

# Projeto 4 (refinado): Segmentação de Tumor em Imagem de Ressonância Magnética
# Melhora da máscara por binarização manual, equalização de histograma e filtragem por área.

# 1) Carrega a imagem
imagem = cv2.imread('Tumor (1).jpg')  # ajuste o nome se necessário
if imagem is None:
    raise FileNotFoundError("Não foi possível carregar a imagem. Verifique o caminho.")

# 2) Converte para escala de cinza e equaliza contraste
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cinza_equalizado = cv2.equalizeHist(cinza)

# 3) Binarização: mantém apenas píxeis muito claros (tumor brilhante)
#    Ajuste o limiar (200) conforme contraste da sua imagem
_, thresh = cv2.threshold(cinza_equalizado, 200, 255, cv2.THRESH_BINARY)

# 4) Limpeza morfológica: remove ruídos pequenos e preenche buracos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# Abre (remove regiões pequenas)
sem_ruido = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# Fecha (preenche fendas internas)
sem_ruido = cv2.morphologyEx(sem_ruido, cv2.MORPH_CLOSE, kernel, iterations=2)

# 5) Filtragem de componentes por área: mantém só regiões grandes
contorno, _ = cv2.findContours(sem_ruido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
area_min = 500  # área mínima do tumor (ajuste se necessário)
mascara = np.zeros_like(sem_ruido)
contornos_grand = [c for c in contorno if cv2.contourArea(c) > area_min]
cv2.drawContours(mascara, contornos_grand, -1, 255, thickness=cv2.FILLED)

# 6) Gera imagens finais: máscara, região segmentada e overlay
segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara)
overlay = imagem.copy()
cv2.drawContours(overlay, contornos_grand, -1, (0,0,255), 2)

# 7) Transforma para RGB
cinza_equalizado = cv2.cvtColor(cinza_equalizado, cv2.COLOR_BGR2RGB)
mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2RGB)
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# 8) Exibe resultados lado a lado
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cinza_equalizado)
plt.title('Cinza Equalizada')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mascara)
plt.title('Máscara')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title('Overlay Contorno Tumor')
plt.axis('off')

plt.tight_layout()
plt.show()


