import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten
import numpy as np
import cv2

class BrainTumorClassifier:
    def __init__(self):
        self.train_images = []
        self.validation_images = []
        self.test_images = []
        self.train_labels = []
        self.validation_labels = []

    def load_images_and_labels(self, data_dir):
        for index, filename in enumerate(os.listdir(data_dir)):
            image = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=-1)
            if index < 15000:
                self.train_images.append(image)
            elif index < 17000:
                self.validation_images.append(image)
            else:
                self.test_images.append(image)

    def load_labels(self, train_labels_file, validation_labels_file):
        with open(train_labels_file) as f:
            next(f)
            self.train_labels = [int(line.split(',')[1]) for line in f]
        with open(validation_labels_file) as f:
            next(f)
            self.validation_labels = [int(line.split(',')[1]) for line in f]

    def preprocess_data(self):
        self.train_images = np.array(self.train_images)
        self.validation_images = np.array(self.validation_images)
        self.test_images = np.array(self.test_images)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    def train_model(self, train_data_gen, validation_data_gen, epochs=150):
        model = self.create_model()
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.optimizers.Adam(),
            metrics=['accuracy', Precision(), Recall()]
        )

        best_model_checkpoint = ModelCheckpoint(
            'bestModel.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        )

        model.fit(train_data_gen, epochs=epochs, validation_data=validation_data_gen, callbacks=[best_model_checkpoint])

    def predict_test_data(self, test_data_gen):
        model = self.create_model()
        model.load_weights('bestModel.h5')
        test_labels = model.predict(test_data_gen)
        predicted_test_labels = np.round_(test_labels)
        return predicted_test_labels

    def save_submission_csv(self, predicted_test_labels, output_file):
        with open(output_file, 'w') as f:
            f.write('id,class\n')
            for index, filename in enumerate(os.listdir('../data/data')):
                if index < 17000:
                    continue
                image_id = filename.split('.')[0]
                predicted_class = int(predicted_test_labels[index - 17000])
                f.write(f'{image_id},{predicted_class}\n')

brain_tumor_classifier = BrainTumorClassifier()

current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, 'data', 'data')
train_labels_file = os.path.join(current_dir, 'data', 'train_labels.txt')
validation_labels_file = os.path.join(current_dir, 'data', 'validation_labels.txt')
submission_csv_file = os.path.join(current_dir, 'data', 'sample_submission.csv')

brain_tumor_classifier.load_images_and_labels(data_dir)
brain_tumor_classifier.load_labels(train_labels_file, validation_labels_file)
brain_tumor_classifier.preprocess_data()

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_data_iterator = train_data_gen.flow(
    brain_tumor_classifier.train_images,
    brain_tumor_classifier.train_labels,
    batch_size=32,
    shuffle=True,
)

validation_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_iterator = validation_data_gen.flow(
    brain_tumor_classifier.validation_images,
    brain_tumor_classifier.validation_labels,
    batch_size=32,
    shuffle=False,
)

test_data_gen = ImageDataGenerator(rescale=1./255)
test_data_iterator = test_data_gen.flow(
    brain_tumor_classifier.test_images,
    batch_size=32,
    shuffle=False,
)

brain_tumor_classifier.train_model(train_data_iterator, validation_data_iterator)
predicted_test_labels = brain_tumor_classifier.predict_test_data(test_data_iterator)
brain_tumor_classifier.save_submission_csv(predicted_test_labels, submission_csv_file)


