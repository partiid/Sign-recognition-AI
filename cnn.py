import numpy as np
import matplotlib.pyplot as plt 
import os
import pandas as pd
import random 

from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical
import cv2 

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ================ params ==========
dataPath = './myData'
labels = './labels.csv'
batch_size = 50
steps_per_epoch = 1650
number_epochs = 15 #number of epochs
tensor_size = (32, 32, 3)

split_ratio = 0.2
validation_ratio = 0.2


# import images 

images_count = 0

images = []
class_idx = [] # classes index

myList = os.listdir(dataPath)
classes_count = len(myList)


print(f"Detected {classes_count} classes")
print('Importing classes....')

for x in range(0, classes_count):
    imagesList = os.listdir(dataPath+'/'+str(images_count))
    for y in imagesList:
        image = cv2.imread(dataPath+'/'+str(images_count)+'/'+y) #read image 
        images.append(image)
        class_idx.append(images_count)
    print(images_count, end=" ")
    images_count += 1
print(" ")
#convert to np array
images = np.array(images)
class_idx = np.array(class_idx)


#split the data 
x_train, x_test, y_train, y_test = train_test_split(images, class_idx, test_size=split_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

# x - images to train 
# y - class label 
print("Train", end="");print(x_train.shape,y_train.shape)
print("validation", end="");print(x_validation.shape,y_validation.shape)
print("test", end="");print(x_test.shape,y_test.shape)


print("Data shapes")
data = pd.read_csv(labels)
print(f"data shape: {data.shape}, {type(data)}")


#display samples 

samples_num = []; 
cols = 5 
num_classess = classes_count
fig,axs = plt.subplots(nrows=num_classess, ncols=cols, figsize = (5,300))
fig.tight_layout()

for i in range(0, cols):
    for j,row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis('off')
        if i == 2:
            axs[j][i].set_title(str(j) + " - " + row["Name"])
            samples_num.append(len(x_selected))

print(f"Number of samples: {samples_num}")
plt.figure(figsize=(12,4))
plt.bar(range(0, classes_count), samples_num)
plt.title("distribution of samples in dataset")
plt.xlabel("class number")
plt.ylabel('number of images')
plt.show()

# ====== preprocessing for better results ======= 


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img



x_train = np.array(list(map(preprocess, x_train)))
x_validation = np.array(list(map(preprocess, x_validation)))
x_test = np.array(list(map(preprocess, x_test)))

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_validation=x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, 
    height_shift_range=0.1,
    zoom_range = 0.2,
    shear_range = 0.1,
    rotation_range=10
)

dataGen.fit(x_train)
batches = dataGen.flow(x_train, y_train, batch_size=batch_size)
x_batch, y_batch = next(batches)


# cnn model 
y_train = to_categorical(y_train, classes_count)
y_validation = to_categorical(y_validation, classes_count)
y_test = to_categorical(y_test, classes_count)

def cnnModel():
    filters_count = 60 
    filter_size = (5,5)

    filter2_size = (3,3)
    pool_size = (2,2)
    nodes_count = 500 
    model = Sequential()
    model.add((Conv2D(filters_count, filter_size, input_shape=(tensor_size[0], tensor_size[1], 1), activation='relu')))
    model.add((Conv2D(filters_count, filter_size, activation='relu')))
    model.add((MaxPooling2D(pool_size=pool_size)))


    model.add((Conv2D(filters_count//2, filter2_size, activation='relu')))
    model.add((Conv2D(filters_count//2, filter2_size, activation='relu')))
    model.add((MaxPooling2D(pool_size=pool_size)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nodes_count, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_count, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = cnnModel()

print(model.summary())
history = model.fit(dataGen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=936, epochs=number_epochs, validation_data=(x_validation, y_validation))

print('Saving model to json...')
json_model = model.to_json()
with open('cnn_model.json', 'w') as json_file:
    json_file.write(json_model)


print('Saving model to h5')
model.save('cnn_model.h5')


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

