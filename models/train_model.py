import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------- PATHS ----------------
PSG_AUDIO_PATH = "../data/archive/PSG-AUDIO/APNEA_EDF"
APNEA_TYPES_PATH = "../data/archive/APNEA_types"

# ---------------- PARAMETERS ----------------
chunk_size = 3000
batch_size = 16
epochs = 10

# ---------------- FILE LISTS ----------------
psg_folders = sorted([os.path.join(PSG_AUDIO_PATH, f) for f in os.listdir(PSG_AUDIO_PATH)
                      if os.path.isdir(os.path.join(PSG_AUDIO_PATH, f))])
label_files = sorted([os.path.join(APNEA_TYPES_PATH, f) for f in os.listdir(APNEA_TYPES_PATH)
                      if f.endswith('.npy')])

if len(psg_folders) != len(label_files):
    raise ValueError("Number of PSG folders does not match number of label files!")

# ---------------- DETECT ALL LABELS ----------------
all_labels_set = set()
for label_file in label_files:
    labels = np.load(label_file)
    all_labels_set.update(labels)
all_labels = sorted(list(all_labels_set))
num_classes = len(all_labels)
label_map = {v: i for i, v in enumerate(all_labels)}  # map original labels to 0..num_classes-1

print(f"Detected {num_classes} classes: {all_labels}")

# ---------------- BATCH GENERATOR ----------------
def batch_generator(psg_folders, label_files, chunk_size=chunk_size, batch_size=batch_size):
    while True:
        for folder, label_file in zip(psg_folders, label_files):
            # Load labels and remap to 0..num_classes-1
            labels = np.load(label_file)
            labels = np.array([label_map[v] for v in labels], dtype=np.int32)

            # Load PSG files in folder
            psg_chunks = []
            for f in os.listdir(folder):
                if f.endswith('.npy'):
                    data = np.load(os.path.join(folder, f), mmap_mode='r')
                    if data.ndim == 1:
                        for i in range(0, len(data) - chunk_size + 1, chunk_size):
                            psg_chunks.append(data[i:i+chunk_size])
                    elif data.ndim == 2:
                        for row in data:
                            if len(row) >= chunk_size:
                                psg_chunks.append(row[:chunk_size])

            # Match labels to chunks
            min_len = min(len(psg_chunks), len(labels))
            psg_chunks = psg_chunks[:min_len]
            labels = labels[:min_len]

            # Yield in batches
            for i in range(0, min_len, batch_size):
                X_batch = np.array(psg_chunks[i:i+batch_size], dtype=np.float32).reshape(-1, chunk_size, 1)
                y_batch = np.array(labels[i:i+batch_size], dtype=np.int32)
                yield X_batch, y_batch

# ---------------- COMPUTE STEPS PER EPOCH ----------------
total_chunks = 0
for folder in psg_folders:
    for f in os.listdir(folder):
        if f.endswith('.npy'):
            data = np.load(os.path.join(folder, f), mmap_mode='r')
            if data.ndim == 1:
                total_chunks += len(data) // chunk_size
            elif data.ndim == 2:
                total_chunks += data.shape[0]
steps_per_epoch = max(total_chunks // batch_size, 1)

print(f"Total chunks: {total_chunks}, steps per epoch: {steps_per_epoch}")

# ---------------- MODEL ----------------
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(chunk_size, 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting training...")

# ---------------- TRAIN ----------------
model.fit(
    batch_generator(psg_folders, label_files, chunk_size=chunk_size, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)

# ---------------- SAVE MODEL ----------------
model.save("sleep_apnea_model.h5")
print("Model saved successfully!")
