from PIL import Image
import os
import pandas as pd 
import numpy as np
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
from tensorflow.keras.utils import plot_model

# Splitting data
from sklearn.model_selection import train_test_split

# Metrics 
from sklearn.metrics import confusion_matrix, classification_report

# Deep Learning
import tensorflow as tf
print('TensoFlow Version: ', tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

path = "./myData"

lab = pd.read_csv("./labels.csv")
i=0
# r=root, d=directories, f = files
for root, directory, file in os.walk(path):
    for file in file:
        if file.endswith('.png'):
            pat=os.path.join(root, file)
            with Image.open(pat) as im:
                if im.size!=(32, 32):
                    im=im.resize((32, 32),Image.LANCZOS)
                im.save(pat.replace(".png",".jpg"))
            os.remove(pat)
            i+=1
            print(i,end='\r')
        elif file.endswith('.jpg'):
            pat=os.path.join(root, file)
            with Image.open(pat) as im:
                
                if im.size!=(32, 32):
                    im=im.resize((32, 32),Image.LANCZOS)
                    im.save(pat)
                    i+=1
                    print(i,end='\r')

# Count PLot of the samples/observations w.r.t the classes
directory = dict()
class_labels = dict()
for dirs in os.listdir(path):
    count = len(os.listdir('./myData/'+dirs))
    directory[dirs+' => '+lab[lab.ClassId == int(dirs)].values[0][1]] = count
    class_labels[int(dirs)] = lab[lab.ClassId == int(dirs)].values[0][1]

plt.figure(figsize = (20, 50))
sns.barplot(y = list(directory.keys()), x = list(directory.values()), palette = 'Set3')
plt.ylabel('Label')
plt.xlabel('Count of Samples/Observations');
plt.show()


#check input data 
# input image dimensions
img_rows, img_cols = 32, 32
# The images are RGB.
img_channels = 3
nb_classes = len(class_labels.keys())

datagen = ImageDataGenerator()
data = datagen.flow_from_directory(path,
                                    target_size=(32, 32),
                                    batch_size=73139,
                                    class_mode='categorical',
                                    shuffle=True )

X , y = data.next()
print(f"Data Shape   :{X.shape}\nLabels shape :{y.shape}")   

#sneek of a data sample 
# fig, axes = plt.subplots(10,10, figsize=(18,18))
# for i,ax in enumerate(axes.flat):
#     r = np.random.randint(X.shape[0])
#     ax.imshow(X[r].astype('uint8'))
#     ax.grid(False)
#     ax.axis('off')
#     ax.set_title('Label: '+str(np.argmax(y[r])))
#plt.show()


# Split the data into training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
print("Train Shape: {}\nTest Shape : {}".format(X_train.shape, X_test.shape))


resnet = ResNet50(weights= None, include_top=False, input_shape= (img_rows,img_cols,img_channels))
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes, activation= 'softmax')(x)
model = Model(inputs = resnet.input, outputs = predictions)


print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
model_check = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

csv_logger = CSVLogger('train_log.csv', separator=',')


n_epochs = 10
history =  model.fit(X_train, y_train,  batch_size = 32, epochs = n_epochs, verbose = 1, 
              validation_data = (X_test, y_test), callbacks = [model_check, early, reduce_lr, csv_logger])

print('Saving model as h5...')
model.save('TSC_model.h5')

print('Saving model as json...')
json_model = model.to_json()
with open('TSC_json_model.json', 'w') as json_file:
    json_file.write(json_model)


loss, acc = model.evaluate(X_test, y_test)
print('Accuracy: ', acc, '\nLoss    : ', loss)


q = len(list(history.history['loss']))
plt.figure(figsize=(12, 6))
sns.lineplot(x = range(1, 1+q), y = history.history['accuracy'], label = 'Accuracy')
sns.lineplot(x = range(1, 1+q), y = history.history['loss'], label = 'Loss')
plt.xlabel('#epochs')
plt.ylabel('Training')
plt.legend();
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x = range(1, 1+q), y = history.history['accuracy'], label = 'Train')
sns.lineplot(x = range(1, 1+q), y = history.history['val_accuracy'], label = 'Validation')
plt.xlabel('#epochs')
plt.ylabel('Accuracy')
plt.legend();

plt.show()