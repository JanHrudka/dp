import cv2
import numpy as np

def detect_colonies(image, template):
    # Převedení obrazů na stupně šedi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Nalezení shody šablony v obraze
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Určení pravého dolního rohu ohraničujícího obdélníku kolem nalezené shody
    h, w = template.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Uložení souřadnic kolonie do seznamu
    colonies = [(top_left, bottom_right)]

    return colonies

# Příklad použití funkce:
# Načtení snímku kolonií mikroorganismů
image = cv2.imread('image.jpg')

# Načtení šablony
template = cv2.imread('template.jpg')

# Detekce kolonií
colonies = detect_colonies(image, template)

# Vypsání souřadnic nalezených kolonií
for colony in colonies:
    print('Kolona na souřadnicích:', colony)