import cv2
import numpy as np
from sklearn import svm

def rozpoznavani_mikrobiologickych_objektu(obrazek, etikety, velikost_trainu=0.8, jadro='rbf', C=1.0):
    # Rozdělení na tréninkovou a testovací množinu
    pocet_objektu = len(etikety)
    pocet_trainu = int(pocet_objektu * velikost_trainu)
    train_data = obrazek[:pocet_trainu].reshape(pocet_trainu, -1)
    train_labels = etikety[:pocet_trainu]
    test_data = obrazek[pocet_trainu:].reshape(pocet_objektu - pocet_trainu, -1)
    test_labels = etikety[pocet_trainu:]

    # Inicializace SVM
    svm_model = svm.SVC(kernel=jadro, C=C)

    # Trénink SVM
    svm_model.fit(train_data, train_labels)

    # Testování SVM
    presnost = svm_model.score(test_data, test_labels)
    print("Presnost klasifikace: {:.2f}%".format(presnost * 100))

    return svm_model