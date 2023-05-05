from PIL import Image, ImageFilter

def adaptive_thresholding(image_path, block_size, c):
    # Načtení obrázku
    img = Image.open(image_path).convert('L')  # Otevření a konverze obrázku do černobílého režimu

    # Aplikace adaptivního prahu
    thresholded = img.filter(ImageFilter.MedianFilter(size=block_size))  # Aplikace mediánového filtru na obrázek
    thresholded = img.point(lambda x: x - c if x > c else x)  # Odečtení konstanty c od hodnot pixelů

    return thresholded

# Cesta k obrázku, velikost bloku a hodnota konstanty
image_path = 'mikroorganismy.jpg'
block_size = 11
c = 7

# Volání funkce pro adaptabilní prahování
thresholded = adaptive_thresholding(image_path, block_size, c)

# Zobrazení původního obrázku
original_image = Image.open(image_path)
original_image.show()

# Zobrazení obrázku po aplikaci prahu
thresholded.show()
