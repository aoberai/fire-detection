import tensorflow as tf
import numpy as np
import os
import cv2

# image = cv2.imread("./fire_dataset/non_fire_images/non_fire.189.png")
#
# image = cv2.resize(src=image, dsize=(150, 150))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#

fire_images_dataset_length = len(os.listdir("./fire_dataset/fire_images"))
non_fire_images_dataset_length = len(os.listdir("./fire_dataset/non_fire_images"))

input_shape = (150, 150, 3)

one_hot_encoded_list = [1] * fire_images_dataset_length + [
    0] * non_fire_images_dataset_length  # 1 represents fire, 0 represents no fire

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#pre-process images
dataset_path = "./fire_dataset/"
fire_images_dataset_dir = os.listdir(dataset_path + "fire_images")
non_fire_images_dataset_dir = os.listdir(dataset_path + "non_fire_images")

fire_detection_train_dataset = []

for image_file_name in fire_images_dataset_dir:
    image = cv2.imread(dataset_path + "fire_images/" + image_file_name)
    if image is None:
        print(dataset_path + "fire_images/" + image_file_name)
    image = cv2.resize(src=image, dsize=(input_shape[0], input_shape[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(np.shape(image))
    fire_detection_train_dataset.append(image)

for image_file_name in non_fire_images_dataset_dir:
    image = cv2.imread(dataset_path + "non_fire_images/" + image_file_name)
    if image is None:
        print(dataset_path + "non_fire_images/" + image_file_name)
    image = cv2.resize(src=image, dsize=(input_shape[0], input_shape[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fire_detection_train_dataset.append(image)

print("\n\nDone loading dataset images.")

print("Training model ...\n\n")

model.fit(x=fire_detection_train_dataset,
          y=one_hot_encoded_list,
          epochs=5)

print("model created")
model_path = "./fire_detection_model.h5"
model.save(model_path)
#
#
# image = cv2.imread("./fire_dataset/non_fire_images/non_fire.11.png")
# image = cv2.resize(src=image, dsize=(150, 150))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# print(image)


# cv2.imshow("test", image)
# cv2.waitKey(0)



# cv2.resize()
