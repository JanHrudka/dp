import cv2

def rozpoznavani_SIFT(img1, img2):
    # Načtení obrázků
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Inicializace detektoru a popisovače SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # Nalezení klíčových příznaků a jejich popisů pro oba obrázky
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Matcher pro porovnávání klíčových příznaků
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Porovnání klíčových příznaků mezi oběma obrázky
    matches = flann.knnMatch(des1, des2, k=2)

    # Filtrace shodujících se příznaků na základě vzájemného poměru vzdáleností
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Vykreslení shodujících se příznaků
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

    # Počet shodujících se příznaků
    pocet_shodujicich_se_priznaku = len(good_matches)

    # Výstup
    print("Počet shodujících se příznaků: ", pocet_shodujicich_se_priznaku)

    # Vykreslení výsledku
    cv2.imshow('Výsledek', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Příklad použití funkce
img1 = cv2.imread('obrazek1.jpg')
img2 = cv2.imread('obrazek2.jpg')
rozpoznavani_SIFT(img1, img2)