import tensorflow as tf

data = tf.keras.datasets.fashion_mnist  # skrot pozwalajacy na dostep do danych

(training_images, training_labels), (test_images, test_labels) = data.load_data()  # zaladowanie zbiorow test i tren
training_images = training_images / 255.0  # podzielenie przez 255 - nomalizacja danych
test_images = test_images / 255.0  # jw. normalizacja poprawia wydajnosc

model = tf.keras.models.Sequential([  # definicja sieci nuronowej, ktora tworzy model
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # chce wyswietlic raport o dokladnosci sieci podczas trenowania
model.fit(training_images, training_labels, epochs=50)  # dopasowanie obrazy tren do etykiet tren w trakcie 5 epok

model.evaluate(test_images, test_labels)  # ocena jakosci modelu

# Flatten - warstwa specyfikacji wejsciowej. wczytujemy tablice 2w 28x28 i zaminiamy ja na 1w
# Dense - ukryta warstwa z neuronami, definiujemy, ze bedzie zawierac 128 neuronow
# w tej warstwie jest tez funkcja aktywacji, czyli kod, ktory zostanie wykonany dla kazdego neuronu
# relu to rektyfiktor, czyli jesli x<=0 => 0, gdy wieksza od 0 to zwraca niezmodyfikowana
# nizej kolejna warstwa Dense, ktora jest warstwa wyjsciowa. ma 10 neuronow, bo mam 10 klas
# kazdy z nich zwroci prawdopod, ze pikse.e pasuja do tej jednej klasy
# funkcja straty jest rzadka kategoryczna entropia krzyżowa
# optymalizator adam to bardziej wydajna wersja sgd.

classifications = model.predict(test_images)
print(classifications[4])
print(test_labels[4])
