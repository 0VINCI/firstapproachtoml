import tensorflow as ts
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# definicja sieci o jednej warstwie z jednym neuronem
# Sequental - definicja warstwy
# Dense - typ warstwy - gesta, tj neurony sa ze soba gesto polaczone
# units=1 = cala siec wykorzystuje tylko 1 neuron
# input_shape=[1] - ksztalt danych wejsciowych, tu pojedyncze liczby
# funkcja straty - komputer okresla, jak dobrze sobie zalozyl rozwiazanie
# sgd - stochastyczny spadek wzdłuż gradientu - optymalizator. Majac poprzednia porgnoze i wartosc bledu
# (lub straty), moze wygenerowac kolejna.
# Zadaniem optymalizatora jest minimalizowanie funkcji straty

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=600)
print(model.predict([10.0]))
print("Oto, czego się nauczylem: {}".format(l0.get_weights()))

# epochs - liczba epok, czyli prob dopasowania