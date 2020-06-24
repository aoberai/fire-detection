import tensorflow as tf
import numpy as np
import os
import cv2


def preprocess_image(preprocessed_image: np.ndarray) -> np.ndarray:
    new_image = cv2.resize(src=preprocessed_image, dsize=(input_shape[0], input_shape[1]))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image


fire_images_dataset_length = len(os.listdir("./fire_dataset/fire_images"))
non_fire_images_dataset_length = len(os.listdir("./fire_dataset/non_fire_images"))


input_shape = (150, 150, 3)

one_hot_encoded_array = np.array([1] * fire_images_dataset_length + [
    0] * non_fire_images_dataset_length)  # 1 represents fire, 0 represents no fire

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# pre-process images
dataset_path = "./fire_dataset/"
fire_images_dataset_dir = os.listdir(dataset_path + "fire_images")
non_fire_images_dataset_dir = os.listdir(dataset_path + "non_fire_images")

fire_detection_train_dataset = []

for image_file_name in fire_images_dataset_dir:
    image = cv2.imread(dataset_path + "fire_images/" + image_file_name)
    new_image = preprocess_image(image)
    fire_detection_train_dataset.append(new_image)

for image_file_name in non_fire_images_dataset_dir:
    image = cv2.imread(dataset_path + "non_fire_images/" + image_file_name)
    new_image = preprocess_image(image)
    fire_detection_train_dataset.append(new_image)

print("\n\nDone loading dataset images.")

print("Training model ...\n\n")

model.fit(x=np.array(fire_detection_train_dataset),
          y=one_hot_encoded_array,
          epochs=10)

print("\n Model created\n\n")
model_path = "./fire_detection_model.h5"
model.save(model_path)

test_image = preprocess_image(cv2.imread("model_test_images/fire.6.png"))
test_image = np.expand_dims(test_image, 0)

fire_threshold = 0.8

prediction = model.predict(test_image)

if prediction[0][0] < prediction[0][1] and prediction[0][1] > fire_threshold:
    print("Image is detected as fire")
    print("Confidence: {0}".format(prediction[0][1] * 100))
else:
    print("Image is detected as not fire")
    print("Confidence: {0}".format(prediction[0][0] * 100))


