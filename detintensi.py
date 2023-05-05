import numpy as np
from PIL import Image

def detect_edges(image_path):
    # Načtení obrázku
    image = Image.open(image_path)
    # Převedení obrázku na černobílý
    grayscale_image = image.convert("L")
    # Konverze obrázku na numpy pole
    img = np.array(grayscale_image)
    
    # Vytvoření Laplaceova jádra pro výpočet druhé derivace
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    
    # Inicializace výsledného obrazu
    edges = np.zeros_like(img)
    
    # Výpočet druhé derivace
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            # Výpočet Laplaceova operátoru
            laplacian_value = np.sum(img[y-1:y+2, x-1:x+2] * laplacian_kernel)
            # Práhování
            if laplacian_value > 0:
                edges[y, x] = 255
    
    # Vytvoření a vrácení výsledného obrazu
    edges_image = Image.fromarray(edges.astype(np.uint8))
    return edges_image
