import cv2
import numpy as np
import matplotlib.pyplot as plt

# Projeto 2: Detecção de Círculos em Imagens
# Detecta ~12 círculos em circulos_1.png usando HoughCircles com parâmetros ajustados.

# 1) Carrega e valida a imagem
imagem_circulos = cv2.imread('circulos_1.png')
if imagem_circulos is None:
    raise FileNotFoundError("Não foi possível carregar 'circulos_1.png'")

# 2) Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem_circulos, cv2.COLOR_BGR2GRAY)

# 3) Suaviza ruído mantendo bordas
#    Kernel menor para não borrar demais o formato circular
suavizado = cv2.GaussianBlur(imagem_cinza, (5, 5), sigmaX=1.5)

# 4) Detecta círculos: Hough Gradient
#    - dp=1.0       resolução do acumulador igual à imagem
#    - minDist=80   distância mínima entre centros (ajuste conforme espaçamento)
#    - param1=100   limiar alto do Canny interno
#    - param2=45    limiar do acumulador (quanto maior, menos círculos espúrios)
#    - minRadius=40 raio mínimo estimado (px)
#    - maxRadius=80 raio máximo estimado (px)
circulos = cv2.HoughCircles(
    suavizado,
    cv2.HOUGH_GRADIENT,
    dp=1.0,
    minDist=80,
    param1=100,
    param2=45,
    minRadius=40,
    maxRadius=80
)

# 5) Se encontrou círculos, converte e desenha
saida = imagem_circulos.copy()
count = 0

if circulos is not None:
    circulos = np.round(circulos[0]).astype(int)
    count = len(circulos)
    for (x, y, r) in circulos:
        # contorno do círculo em verde
        cv2.circle(saida, (x, y), r, (0, 255, 0), 2)
        # centro em vermelho
        cv2.circle(saida, (x, y), 2, (0, 0, 255), 3)

# 6) Exibe resultado com contagem
saida_RGB = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(saida_RGB)
plt.title(f"Círculos detectados: {count}")
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"Total de círculos detectados: {count}")
