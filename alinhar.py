import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import sys

def objetive_function(img1, img2):
    return np.sum((img1 - img2) ** 2)

def align_images_horizontal_projection(img1, img2_path):
    n = img1.shape[0]
    profile = []
    for _ in range(n):
        profile.append(np.sum(img1, axis=1))
    value = [0]*180
    for angle in range(-90, 90):
        value[angle+90].append(objetive_function(profile, np.roll(profile, 1)))
    angle = np.argmax(value) - 90
    print(angle)
    return 0

def align_images_hough(img1, img2_path):

    return 0

def main():
    # Read arguments
    parser = argparse.ArgumentParser(description='Alinhar imagens')
    
    # Add arguments
    parser.add_argument('input_image', type=str, help='Caminho da imagem entrada')
    parser.add_argument('output_image', type=str, help='Caminho da imagem saida')
    parser.add_argument('--method', type=str, default='horizontal_projection', help='Método de alinhamento')

    # Parse arguments
    args = parser.parse_args()
    if args.input_image is None or args.output_image is None:
        print("Você precisa do caminho de uma imagem para alinhar e o caminho para a imagem saida")
        return 0
    
    # Open Image
    img = cv.imread(args.input_image, cv.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if img is None:
        sys.exit("Não foi possível carregar a imagem.")

    # Align images
    if args.method == 'horizontal_projection':
        align_images_horizontal_projection(img, args.output_image)
    elif args.method == 'hough':
        align_images_hough(img, args.output_image)
    return 0

main()