import cv2

def detect_and_describe_SURF(image):
    # Převedení obrázku na stupně šedi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Inicializace objektu SURF
    surf = cv2.xfeatures2d.SURF_create()

    # Detekce klíčových bodů a jejich popis
    keypoints, descriptors = surf.detectAndCompute(gray, None)

    # Vykreslení klíčových bodů na obrázku
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, descriptors, image_with_keypoints