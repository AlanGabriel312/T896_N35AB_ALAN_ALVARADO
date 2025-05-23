import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random

# Carrega o modelo treinado
modelo = YOLO('dataset/runs/detect/train4/weights/best.pt')


# 1) Carrega e valida a imagem
imagem_moedas = cv2.imread('dataset/images/val/IMG_20250521_154631.jpg')
if imagem_moedas is None:
    raise FileNotFoundError("Não foi possível carregar a imagem.")

# 2) Converte para escala de cinza
imagem_cinza = cv2.cvtColor(imagem_moedas, cv2.COLOR_BGR2GRAY)

# 3) Suaviza ruído mantendo bordas
#    Kernel menor para não borrar demais o formato circular
suavizado = cv2.GaussianBlur(imagem_cinza, (5, 5), sigmaX=1.5)

# 4) Detecta círculos: Hough Gradient
#    - dp=1.0       resolução do acumulador igual à imagem
#    - minDist=900  distância mínima entre centros (ajuste conforme espaçamento)
#    - param1=100   limiar alto do Canny interno
#    - param2=50    limiar do acumulador (quanto maior, menos círculos espúrios)
#    - minRadius=400 raio mínimo estimado (px)
#    - maxRadius=500 raio máximo estimado (px)
moedas = cv2.HoughCircles(
    suavizado,
    cv2.HOUGH_GRADIENT,
    dp=1.0,
    minDist=900,
    param1=100,
    param2=50,
    minRadius=400,
    maxRadius=500
)

# 5) Se encontrou círculos, converte e desenha
saida = imagem_moedas.copy()
count = 0

if moedas is not None:
    moedas = np.round(moedas[0]).astype(int)
    count = len(moedas)
    for (x, y, r) in moedas:
        # contorno do círculo em verde
        cv2.circle(saida, (x, y), r, (0, 255, 0), 5)

# 6) Exibe resultado dos círculos detectados
saida_RGB = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(saida_RGB)
plt.title(f"Moedas detectadas: {count}")
plt.axis('off')
plt.tight_layout()
plt.show()

print("=== DETECÇÃO DE MOEDAS ===")
print(f"Total de moedas detectadas: {count}")

# 7) Inicializa estruturas para contagem e valores
contagem = {}

valores_moedas = {
    '5c': 0.05,
    '10c': 0.10,
    '25c': 0.25,
    '50c': 0.50,
    '1real': 1.00
}

valor_total = 0

# 8) Realiza a detecção das moedas com YOLO
resultados = modelo.predict(source=saida, save=False, verbose=False)
resultado = resultados[0]

# 9) Processa as detecções e anota na imagem
for box in resultado.boxes:
    cls_id = int(box.cls)
    nome = resultado.names[cls_id]
    conf = box.conf.item()

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Desenha retângulo na moeda detectada
    cv2.rectangle(saida, (x1, y1), (x2, y2), (0, 255, 0), 6)
    # Coloca label com nome e confiança
    cv2.putText(saida, f'{nome} {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)

    print(f'Moeda detectada: {nome} | Confiança: {conf:.2f}')

    # Atualiza contagem e valor total
    contagem[nome] = contagem.get(nome, 0) + 1
    valor_total += valores_moedas.get(nome, 0)

# 10) Exibe a imagem final com as detecções YOLO
saida_RGB = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(saida_RGB)
plt.title("Resultado Final com Detecção YOLO")
plt.axis('off')
plt.tight_layout()
plt.show()


# 11) Imprime o valor final
print("\n==== VALOR FINAL ====")
for moeda, qtd in contagem.items():
    print(f'{moeda}: {qtd} unidade(s)')

print(f'Valor total detectado na imagem: R$ {valor_total:.2f}')



# === SIMULAÇÃO DE DANO NAS MOEDAS ===

print("\n=== SIMULAÇÃO DE DANO ===")

for (x, y, r) in moedas:
    # 12) Cria uma máscara circular para isolar a moeda
    mascara = np.zeros(imagem_cinza.shape, dtype=np.uint8)
    cv2.circle(mascara, (x, y), r, 255, -1)  # Círculo preenchido da moeda

    # 13) Associa a moeda detectada pelo Hough com a detectada pelo YOLO
    # Busca a caixa YOLO mais próxima do centro da moeda (x, y)
    nome_moeda = 'Moeda Não Detectada'  # Padrão caso não encontre
    menor_dist = float('inf')

    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centro_box = ((x1 + x2) // 2, (y1 + y2) // 2)

        dist = np.sqrt((centro_box[0] - x) ** 2 + (centro_box[1] - y) ** 2)

        if dist < menor_dist:
            menor_dist = dist
            cls_id = int(box.cls)
            nome_moeda = resultado.names[cls_id]

    # 14) Define a quantidade aleatória de manchas (entre 0 e 2)
    quantidade_manchas = random.randint(0, 2)

    for _ in range(quantidade_manchas):
        # 15) Define o raio da mancha como 25% do raio da moeda
        raio_manchas = int(r * 0.25)

        # 16) Gera uma posição aleatória dentro da moeda
        while True:
            offset_x = random.randint(-r + raio_manchas, r - raio_manchas)
            offset_y = random.randint(-r + raio_manchas, r - raio_manchas)

            distancia_ao_centro = (offset_x) ** 2 + (offset_y) ** 2

            if distancia_ao_centro <= (r - raio_manchas) ** 2:
                break  # Se a mancha estiver dentro da moeda, aceita a posição

        centro_manchas = (x + offset_x, y + offset_y)

        # 17) Desenha a mancha preta simulando dano na moeda
        cv2.circle(saida, centro_manchas, raio_manchas, (0, 0, 0), -1)

    # 18) Exibe no terminal o nome da moeda e quantidade de manchas simuladas
    print(f"Adicionadas {quantidade_manchas} manchas simuladas na moeda de {nome_moeda}")


# 17) Converte para RGB pra exibir no matplotlib (OpenCV usa BGR) e printa a imagem
saida_RGB = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(saida_RGB)
plt.title("Imagem com Mancha Simulada nas Moedas")
plt.axis('off')
plt.tight_layout()
plt.show()


# === VERIFICAÇÃO DO ESTADO DAS MOEDAS ===

print("\n=== VERIFICAÇÃO DO ESTADO DAS MOEDAS ===")

# 19) Converte a imagem para HSV para analisar o canal de brilho (V)
imagem_hsv = cv2.cvtColor(saida_RGB, cv2.COLOR_BGR2HSV)
canal_v = imagem_hsv[:, :, 2]  # Extrai o canal V (brilho)

for (x, y, r) in moedas:
    # 20) Associa a moeda detectada pelo Hough com a detectada pelo YOLO
    nome_moeda = 'Moeda Não Detectada'
    menor_dist = float('inf')

    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centro_box = ((x1 + x2) // 2, (y1 + y2) // 2)

        dist = np.sqrt((centro_box[0] - x) ** 2 + (centro_box[1] - y) ** 2)

        if dist < menor_dist:
            menor_dist = dist
            cls_id = int(box.cls)
            nome_moeda = resultado.names[cls_id]

    # 21) Cria uma máscara circular para isolar a moeda
    mascara = np.zeros(imagem_cinza.shape, dtype=np.uint8)
    cv2.circle(mascara, (x, y), r, 255, -1)  # Máscara preenchida da moeda

    # 22) Aplica a máscara no canal V (brilho) para isolar a moeda
    moeda_recorte = cv2.bitwise_and(canal_v, canal_v, mask=mascara)

    # 23) Suaviza a imagem da moeda para reduzir reflexos e ruídos
    moeda_suave = cv2.GaussianBlur(moeda_recorte, (7, 7), 0)

    # 24) Extrai apenas os pixels dentro da máscara (moeda)
    pixels_moeda = moeda_suave[mascara == 255]

    # 25) Calcula o desvio padrão dos pixels (mede a variação de textura da moeda)
    desvio = np.std(pixels_moeda)

    print(f'{nome_moeda} - Desvio Padrão: {desvio:.2f}')

    # 26) Define o estado da moeda com base no desvio
    if desvio > 50:  # Limiar ajustável - Teste no seu dataset
        estado = 'DANIFICADA'
        cor_estado = (0, 0, 255)  # Vermelho
    else:
        estado = 'BOA'
        cor_estado = (0, 255, 0)  # Verde

    # 27) Desenha na imagem o estado da moeda (Boa ou Danificada)
    cv2.putText(saida, estado, (x - r, y + r + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 4, cor_estado, 10)

    print(f'Estado da moeda de {nome_moeda}: {estado}')

# 28) Exibe a imagem final com o estado das moedas anotado
saida_RGB_final = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(saida_RGB_final)
plt.title("Estado Final das Moedas (Boa ou Danificada)")
plt.axis('off')
plt.tight_layout()
plt.show()




