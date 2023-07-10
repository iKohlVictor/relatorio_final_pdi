import cv2
import numpy as np
import matplotlib.pyplot as plt

# funções


def get_rgb_channels(imagem):
    # Divide a imagem em canais de cores (RGB)
    canais = cv2.split(imagem)

    # Obtém os canais individuais
    blue_channel = canais[0]
    green_channel = canais[1]
    red_channel = canais[2]

    return blue_channel, green_channel, red_channel


def get_hsv_channels(imagem):
    # Converte a imagem para o espaço de cores HSV
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Separa as bandas HSV
    hue_channel = imagem_hsv[:, :, 0]
    saturation_channel = imagem_hsv[:, :, 1]
    value_channel = imagem_hsv[:, :, 2]

    return imagem_hsv, hue_channel, saturation_channel, value_channel


def show_image(imagem, nome_da_janela):
    cv2.namedWindow(nome_da_janela, cv2.WINDOW_NORMAL)
    cv2.imshow(nome_da_janela, imagem)


def get_histograma(imagem):
    # Calcula o histograma completo
    histograma = cv2.calcHist([imagem], [0], None, [256], [0, 256])

    # Converte a imagem para escala de cinza
    imagem_em_escala_de_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Equaliza o histograma
    imagem_equalizada = cv2.equalizeHist(imagem_em_escala_de_cinza)

    # Calcula o histograma equalizado
    histograma_equalizado = cv2.calcHist(
        [imagem_equalizada], [0], None, [256], [0, 256])

    return histograma, histograma_equalizado


def show_histograma(histograma, plot, nomeDaJanela):
    plt.figure()
    plt.subplot(plot[0], plot[1], plot[2])
    plt.plot(histograma)
    plt.title(nomeDaJanela)
    plt.xlabel("Níveis de Cinza")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()


# programa principal
nomeDaImagem = "C.01_T.1_P.0001.jpg"

# Carrega a imagem

caminhoDaImagem = ("./LuzBranca_Baixo/" + nomeDaImagem)

print(caminhoDaImagem)

imagem = cv2.imread(caminhoDaImagem)


# Etapas.

[blue_channel, green_channel, red_channel] = get_rgb_channels(imagem)

[imagem_hsv, hue_channel, saturation_channel,
    value_channel] = get_hsv_channels(imagem)

show_image(imagem, "Imagem Original")
show_image(imagem_hsv, "Imagem HSV")
show_image(value_channel, "Banda V")

# histograma da imagem original
# histograma, histograma_equalizado = get_histograma(imagem)
# show_histograma(histograma, [2, 1, 1], "Histograma da Imagem Original")
# show_histograma(histograma_equalizado, [2, 1, 2], "Histograma Equalizado")


# 2 - fazer limiarização de uma banda apenas

# Seleciona a banda desejada (por exemplo, a banda verde)

# banda_escolhida = hue_channel

# Aplica a limiarização na banda escolhida

# limiar, imagem_limiarizada = cv2.threshold(
#     banda_escolhida, 128, 255, cv2.THRESH_BINARY)


# cv2.namedWindow("Limiarizada", cv2.WINDOW_NORMAL)
# cv2.imshow("Limiarizada", imagem_limiarizada)

# # 3 - aplicar operações morfológicas para remover ruídos

# # Aplica a operação de abertura para remover ruído

# elemento_estruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# imagem_limpa = cv2.morphologyEx(
#     imagem_limiarizada, cv2.MORPH_OPEN, elemento_estruturante)

# cv2.namedWindow("Imagem Limpa", cv2.WINDOW_NORMAL)
# cv2.imshow("Imagem Limpa", imagem_limpa)

# 4 - multiplicar pela a imagem original para fazer uma espécie de máscara
# 5 - definir um centro de massa da imagem e recortar um retângulo
# 6 - extrair valores de variância


# Espera pela tecla de fechar a janela
while True:
    if cv2.waitKey(1) == ord('q'):  # Verifica se a tecla 'q' foi pressionada
        break

cv2.destroyAllWindows()


# Mostra a imagem equalizada
cv2.waitKey(0)
cv2.destroyAllWindows()
