import cv2
# define uma função que recebe uma imagem e retorna as 3 bandas RGB


def get_rgb_channels(imagem):
    # Divide a imagem em canais de cores (RGB)
    canais = cv2.split(imagem)

    # Obtém os canais individuais
    blue_channel = canais[0]
    green_channel = canais[1]
    red_channel = canais[2]

    return blue_channel, green_channel, red_channel

# Define uma função que recebe uma imagem e retorna as 3 bandas HSV


def get_hsv_channels(imagem):
    # Converte a imagem para o espaço de cores HSV
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Separa as bandas HSV
    hue_channel = imagem_hsv[:, :, 0]
    saturation_channel = imagem_hsv[:, :, 1]
    value_channel = imagem_hsv[:, :, 2]

    return hue_channel, saturation_channel, value_channel
