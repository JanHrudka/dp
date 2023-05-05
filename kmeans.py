import numpy as np

def kmeans(image, k=3, max_iters=10):
    # Získání rozměrů obrázku
    height, width = image.shape[:2]
    
    # Převod obrázku na 1D pole hodnot pixelů
    if image.ndim == 3:
        image = image.reshape(height * width, -1)
    else:
        image = image.reshape(height * width, 1)
    
    # Inicializace centroidů
    centroids = image[np.random.choice(height * width, k, replace=False)]
    
    # Cyklus pro provedení K-means algoritmu
    for _ in range(max_iters):
        # Výpočet vzdáleností pixelů od centroidů
        distances = np.linalg.norm(image - centroids[:, np.newaxis], axis=2)
        
        # Přiřazení pixelů do skupin
        labels = np.argmin(distances, axis=0)
        
        # Aktualizace hodnot centroidů
        new_centroids = np.array([image[labels == i].mean(axis=0) for i in range(k)])
        
        # Kontrola konvergence
        if np.linalg.norm(new_centroids - centroids) < 1e-4:
            break
        
        centroids = new_centroids
    
    return centroids, labels