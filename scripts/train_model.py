import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ----------------- CONFIG ----------------- #
DATA_PATH = os.path.join('signlanguage', 'data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30
MODEL_PATH = os.path.join('signlanguage', 'model', 'action.h5')
# ------------------------------------------ #

# Label map
label_map = {label: num for num, label in enumerate(actions)}

# Load sequences and labels
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(path):
                window.append(np.load(path))
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# ----------------- Build LSTM Model ----------------- #
model = Sequential()
model.add(Input(shape=(sequence_length, 1662)))  # 30 timesteps, 1662 features
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# ----------------- Train the Model ----------------- #
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# ----------------- Save the Model ----------------- #
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")

