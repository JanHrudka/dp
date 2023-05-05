from PIL import Image

def histogram_thresholding(image_path, threshold):
    # Načtení obrázku
    img = Image.open(image_path).convert('L')  # Otevření a konverze obrázku do černobílého režimu

    # Výpočet histogramu
    histogram = img.histogram()

    # Prahování na základě histogramu
    pixel_count = sum(histogram)
    thresholded = img.point(lambda x: 255 if sum(histogram[:x+1]) / pixel_count < threshold else 0)  # Aplikace prahu na základě histogramu

    return thresholded

# Cesta k obrázku a hodnota prahu
image_path = 'mikroorganismy.jpg'
threshold = 0.5  # Hodnota prahu, která se pohybuje mezi 0 a 1, kde 0 znamená černý výstup a 1 bílý výstup

# Volání funkce pro prahování na základě histogramu
thresholded = histogram_thresholding(image_path, threshold)

# Zobrazení původního obrázku
original_image = Image.open(image_path)
original_image.show()

# Zobrazení obrázku po aplikaci prahu
thresholded.show()