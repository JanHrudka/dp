import numpy as np
from PIL import Image

def detect_edges(image_path):
    # Načtení obrázku
    image = Image.open(image_path)
    # Převedení obrázku na černobílý
    grayscale_image = image.convert("L")
    # Konverze obrázku na numpy pole
    img = np.array(grayscale_image)
    
    # Vytvoření konvolučních jader pro výpočet gradientů
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    
    # Inicializace výsledného obrazu
    edges = np.zeros_like(img)
    
    # Výpočet gradientů
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            # Výpočet gradientu X
            gradient_x = np.sum(img[y-1:y+2, x-1:x+2] * sobel_x)
            # Výpočet gradientu Y
            gradient_y = np.sum(img[y-1:y+2, x-1:x+2] * sobel_y)
            # Celkový gradient
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            # Práhování
            if gradient_magnitude > 128:
                edges[y, x] = 255
    
    # Vytvoření a vrácení výsledného obrazu
    edges_image = Image.fromarray(edges.astype(np.uint8))
    return edges_image
