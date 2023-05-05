from PIL import Image

def global_thresholding(image_path, threshold):
    # Načtení obrázku
    img = Image.open(image_path).convert('L')  # Otevření a konverze obrázku do černobílého režimu

    # Získání rozměrů obrázku
    width, height = img.size

    # Vytvoření prázdného obrázku pro výstup
    thresholded = Image.new('L', (width, height), 0)

    # Aplikace globálního prahu
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))  # Získání hodnoty pixelu
            if pixel >= threshold:
                thresholded.putpixel((x, y), 255)  # Nastavení pixelu na bílou, pokud je hodnota pixelu větší nebo rovna prahu

    return thresholded

# Cesta k obrázku a hodnota prahu
image_path = 'mikroorganismy.jpg'
threshold = 128

# Volání funkce pro globální prahování
thresholded = global_thresholding(image_path, threshold)

# Zobrazení původního obrázku
original_image = Image.open(image_path)
original_image.show()

# Zobrazení obrázku po aplikaci prahu
thresholded.show()
