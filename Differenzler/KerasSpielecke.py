from keras import Input, Model
from keras.layers import Dense, LSTM, np

deep_lstm = Input(shape=(None, 2))
dl = Dense(4)(deep_lstm)
dl = LSTM(1)(dl)

model = Model(inputs=deep_lstm, outputs=dl)
model.compile(optimizer='rmsprop', loss="mse")

x = np.array([[[1, 2], [3, 4]], [[4, 3], [2, 1]]])
y = np.array([1, -1])

model.fit(x, y, epochs=1000)

print(model.predict(np.reshape(x[0], (-1, 2, 2))))
