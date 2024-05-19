from ast import arg
from platform import python_branch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import argparse
import sys

def objetive_function(profile, n):
    result = 0
    for i in range(n-1):
        result += (profile[i] - profile[i+1])**2
    return result

def align_images_horizontal_projection(img1, img2_path):
    n = img1.shape[0]
    profiles = np.zeros((180, n))
    angle_results = np.zeros(180)
    for ang in range(180):
        img_angle = rotate(img1, ang-90, reshape=False)
        for i in range(n):
            profiles[ang][i] = np.sum(img_angle[i])
        angle_results[ang] = objetive_function(profiles[ang], n)
    angle = np.argmax(angle_results) - 90

    # Write image rotated
    img_rotated = rotate(img1, angle, reshape=False)
    print(angle)
    cv.imwrite(img2_path, img_rotated)
    return 0

def align_images_hough(img1, img2_path):
    # Detecção de bordas
    edges = cv.Canny(img1, 50, 150, apertureSize=3)

    # Traformada de linha de Hough
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Pegar angulo mais frequente das linhas
    angles = []
    for line in lines:
        theta = line[0]
        angle = np.rad2deg(theta)
        angles.append(angle)
    angle = round(np.median(angles)) - 90
    print(angle)

    # Rotacionar imagem
    img_rotated = rotate(img1, angle, reshape=False)
    cv.imwrite(img2_path, img_rotated)
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