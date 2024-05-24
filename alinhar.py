import numpy as np
import cv2 as cv
from scipy.ndimage import rotate
import argparse
import sys
from joblib import Parallel, delayed


def objetive_function(profile, n):
    #Função objetivo para o alinhaento baseado na projeção horizontal
    result = 0
    for i in range(n-1):
        result += (profile[i] - profile[i+1])**2
    return result

def process_angle(ang, img, n):
    #Processa um ângulo para a função objetivo
    img_angle = rotate(img, ang-90, reshape=False)
    profile = np.zeros(n)
    for i in range(n):
        profile[i] = np.sum(img_angle[i])
    return objetive_function(profile, n)

def align_images_horizontal_projection(img1, img2_path):
    # Realiza o alinhamento baseado na projeção horizontal
    # Get image size
    n = img1.shape[0]
    
    # Get horizontal projection
    img_process = cv.Canny(img1, 100, 200)
    angle_results = Parallel(n_jobs=-1)(delayed(process_angle)(ang, img_process, n) for ang in range(180))
    angle = np.argmax(angle_results) - 90

    # Write image rotated
    img_rotated = rotate(img1, angle, reshape=False)
    print(angle)
    cv.imwrite(img2_path, img_rotated)
    return 0

def align_images_hough(img1, img2_path):
    m = img1.shape[1]
    
    # Detecção de bordas
    edges = cv.Canny(img1, 50, 150, apertureSize=3)

    # Traformada de linha de Hough
    lines = cv.HoughLines(edges, 1, np.pi / 180, (m//6)*2)
    if lines is None:
        sys.exit("Não foi possível encontrar linhas na imagem.")
    
    # Pegar angulo mais frequente das linhas
    angles = []
    for line in lines:
        theta = line[0][1]
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