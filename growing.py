import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import flood, flood_fill


def region_growing(image, seed, threshold):
    height, width = image.shape
    visited = np.zeros((height, width))  # pole pro sledování již navštívených pixelů
    mask = np.zeros((height, width))  # binární maska segmentované oblasti

    # Funkce pro kontrolu platnosti souřadnic
    def is_valid_coord(x, y):
        return 0 <= x < width and 0 <= y < height

    # Získání hodnoty intenzity výchozího bodu
    seed_x, seed_y = seed
    seed_value = image[seed_y, seed_x]

    # Fronta pro uchovávání pixelů k prozkoumání
    queue = [(seed_x, seed_y)]

    # Provádění region growing
    while len(queue) > 0:
        x, y = queue.pop(0)
        visited[y, x] = 1
        mask[y, x] = 1

        # Procházení okolních pixelů
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # vynechání středového pixelu
                nx, ny = x + dx, y + dy
                if is_valid_coord(nx, ny) and not visited[ny, nx]:
                    # Porovnání hodnoty intenzity okolního pixelu s prahovou hodnotou
                    if abs(image[ny, nx] - seed_value) <= threshold:
                        queue.append((nx, ny))
                        visited[ny, nx] = 1

    return mask


# Příklad použití
# Načtení šedotónového obrázku
image = np.array([[10, 10, 10, 30, 30, 30],
                  [10, 10, 10, 30, 30, 30],
                  [10, 10, 10, 30, 30, 30],
                  [30, 30, 30, 10, 10, 10],
                  [30, 30, 30, 10, 10, 10],
                  [30, 30, 30, 10, 10, 10]])

# Zvolení výchozího bodu
seed = (0, 0)

# Nastavení prahu pro rozšiřování regionu
threshold = 20

# Spuštění region growing algoritmu
mask = region_growing(image, seed, threshold)