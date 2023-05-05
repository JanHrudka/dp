import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_microorganism_detection_cnn():
    # Vytvoření modelu CNN
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Pouze jedna výstupní vrstva s aktivací sigmoid pro pravděpodobnost

    # Kompilace modelu
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Funkce pro načtení obrázku a předzpracování pro vstup do CNN
def load_and_preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0  # Normalizace hodnot pixelů na rozsah [0, 1]
    return image

# Funkce pro predikci pravděpodobnosti přítomnosti mikroorganismů v obraze
def predict_microorganism_presence(image_path, model):
    image = load_and_preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction[0][0]  # Výstup je jednočlenný seznam, vrátíme hodnotu pravděpodobnosti

# Příklad použití funkce
model = create_microorganism_detection_cnn()
model.load_weights('model_weights.h5')  # Načtení váhových hodnot modelu
image_path = 'obraz.jpg'  # Cesta k obrázku, který chceme analyzovat
prediction = predict_microorganism_presence(image_path, model)
print(f'Pravděpodobnost přítomnosti mikroorganismů v obraze: {prediction:.2f}')