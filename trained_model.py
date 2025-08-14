import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Set dataset paths
base_dir = "C:\\Users\\haran\\OneDrive\\Desktop\\CODE_ALPHA\\CAREER_NAVIGATOR_AI\\SoftSkills\\SoftSkills\\fer2013_extracted"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')
test_data = datagen.flow_from_directory(test_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=test_data, epochs=10)

# Save the trained model
model.save("C:\\Users\\haran\\OneDrive\\Desktop\\CODE_ALPHA\\CAREER_NAVIGATOR_AI\\SoftSkills\\SoftSkills\\fer_model.h5")
print("✅ Model trained and saved as fer_model.h5")
