import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# funções


def get_rgb_channels(imagem):
    # Divide a imagem em canais de cores (RGB)
    r, g, b = cv2.split(imagem)

    # Obtém os canais individuais
    blue_channel = cv2.merge([b, b, b])
    green_channel = cv2.merge([g, g, g])
    red_channel = cv2.merge([r, r, r])

    return blue_channel, green_channel, red_channel


def get_hsv_channels(imagem):
    # Converte a imagem para o espaço de cores HSV
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(imagem_hsv)

    # Separa as bandas HSV
    hue_channel = cv2.merge([h, h, h])
    saturation_channel = cv2.merge([s, s, s])
    value_channel = cv2.merge([v, v, v])

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


def load_histograma(histograma, nomeDaJanela, xLabel, yLabel):
    plt.plot(histograma)
    plt.title(nomeDaJanela)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('./imagem_convertida/' + pasta + '/' +
                arquivo + '/01-histograma_original.png')


def limiarizar(imagem, limiar, limite_maximo, tipo):
    limiar, imagem_limiarizada = cv2.threshold(
        imagem, limiar, limite_maximo, tipo)

    return imagem_limiarizada


# programa principal
pasta = "LuzBranca_Baixo"
arquivo = "C.01_T.1_P.0001"

nomeDaImagem = "C.01_T.1_P.0001.jpg"


diretorioDestino = './imagem_convertida/' + pasta + '/' + arquivo
if not os.path.exists(diretorioDestino):
    os.makedirs(diretorioDestino)

# Carrega a imagem

caminhoDaImagem = ("./LuzBranca_Baixo/" + nomeDaImagem)


imagem = cv2.imread(caminhoDaImagem)
cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/01-imagem_original.jpg", imagem)

histograma_original = cv2.calcHist([imagem], [0], None, [256], [0, 256])
load_histograma(histograma_original, "Histograma Original",
                "Valores de Pixel", "Frequência")

imagem_to_yuv = cv2.cvtColor(imagem, cv2.COLOR_BGR2YUV)
imagem_to_yuv[:, :, 0] = cv2.equalizeHist(imagem_to_yuv[:, :, 0])
imagem_equalizada = cv2.cvtColor(imagem_to_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite("./imagem_convertida/" + pasta + '/' + arquivo+"/01-imagem_equalizada.jpg",
            imagem_equalizada)

histograma_equalizado = cv2.calcHist(
    [imagem_equalizada], [0], None, [256], [0, 256])

load_histograma(histograma_equalizado, "Histograma Original e Histograma Equalizado",
                'Valores de Pixel', 'Frequência')

# separa os canais de cores
blue_channel, green_channel, red_channel = get_rgb_channels(imagem)

cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/02-imagem_r.jpg", red_channel)
cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/03-imagem_g.jpg", green_channel)
cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/04-imagem_b.jpg", red_channel)


imagem_hsv, hue_channel, saturation_channel, value_channel = get_hsv_channels(
    imagem)

cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/05-imagem_hsv.jpg", imagem_hsv)
cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/06-imagem_h.jpg", hue_channel)
cv2.imwrite("./imagem_convertida/"+pasta + '/' +
            arquivo +
            "/07-imagem_s.jpg", saturation_channel)
cv2.imwrite("./imagem_convertida/" + pasta + '/' +
            arquivo+"/08-imagem_v.jpg", value_channel)


# aplicando o filtro

imagem_filtrada = cv2.GaussianBlur(value_channel, (5, 5), 0)

cv2.imwrite("./imagem_convertida/"+pasta + '/' +
            arquivo +
            "/09-imagem_filtrada.jpg", imagem_filtrada)

# limiarização

imagem_limiarizada = limiarizar(imagem_filtrada, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite("./imagem_convertida/" + pasta + '/' + arquivo+"/10-imagem_limiarizada.jpg",
            imagem_limiarizada)

# morfológicas

kernel = np.ones((5, 5), np.uint8)

imagem_processada = cv2.morphologyEx(
    imagem_limiarizada, cv2.MORPH_OPEN, kernel, iterations=4)
imagem_processada = cv2.morphologyEx(
    imagem_processada, cv2.MORPH_CLOSE, kernel, iterations=4)

cv2.imwrite("./imagem_convertida/"+pasta + '/' +
            arquivo +
            "/11-imagem_dilatada.jpg", imagem_processada)

# multiplicar

imagem_multiplicada = np.multiply(imagem, imagem_processada)

cv2.imwrite("./imagem_convertida/" + pasta + '/' + arquivo+"/12-imagem_multiplicada.jpg",
            imagem_multiplicada)


# Encontrar contornos

imagem_cinza = cv2.cvtColor(imagem_multiplicada, cv2.COLOR_BGR2GRAY)

contornos, _ = cv2.findContours(
    imagem_cinza, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Encontrando o centro de massa

maior_contorno = max(contornos, key=cv2.contourArea)
M = cv2.moments(maior_contorno)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# Desenhando o centro de massa
x = cx - 300 // 2
y = cy - 300 // 2

retangulo_recortado = imagem[max(y, 0):y + 300, max(x, 0):x + 300]

cv2.imwrite("./imagem_convertida/" + pasta + '/' + arquivo+"/13-retangulo_recortado.jpg",
            retangulo_recortado)

variancia = np.var(retangulo_recortado)
media = np.mean(retangulo_recortado)
desvio_padrao = np.std(retangulo_recortado)
mediana = np.median(retangulo_recortado)

print("Variancia: ", variancia)
print("Media: ", media)
print("Desvio Padrão: ", desvio_padrao)
print("Mediana: ", mediana)
