import cv2
import pandas as pd
import numpy as np
import scipy.misc
import os
from random import shuffle
import tensorflow as tf
import matplotlib.image as mpimg
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PIL import Image, ImageDraw, ImageFont

TRAIN_DIR = 'images'
TEST_DIR = 'Timages'
IMG_SIZE =150
LR = 0.0001
MODEL_NAME = 'WeaponsDetection-cnn'

import cv2
import pandas as pd
import numpy as np
import scipy.misc
import os
import matplotlib.image as mpimg
from random import shuffle
import matplotlib.pyplot as plt
import glob
import sklearn.model_selection as sk
#from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# load and display an image with Matplotlib
from matplotlib import image
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
TRAIN_DIR = 'train'
TEST_DIR='.test'
IMG_SIZE = 256
LR = 0.001
MODEL_NAME = 'Scene-recognation-cnn'

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name
    if word_label == 'airport_inside' or 'airport_inside' in word_label:
        return np.array([1,0,0,0,0,0,0,0,0,0])
    elif word_label == 'bakery' or 'bakery' in word_label:
        return np.array([0,1,0,0,0,0,0,0,0,0])
    elif word_label == 'bedroom' or 'bedroom' in word_label:
        return np.array([0,0,1,0,0,0,0,0,0,0])
    elif word_label == 'greenhouse' or 'greenhouse' in word_label:
        return np.array([0,0,0,1,0,0,0,0,0,0])
    elif word_label == 'gym' or 'gym' in word_label:
        return np.array([0,0,0,0,1,0,0,0,0,0])
    elif word_label == 'kitchen' or 'kitchen' in word_label:
        return np.array([0,0,0,0,0,1,0,0,0,0])
    elif word_label == 'operating_room' or 'operating_room' in word_label:
        return np.array([0,0,0,0,0,0,1,0,0,0])
    elif word_label == 'poolinside' or 'poolinside' in word_label:
        return np.array([0,0,0,0,0,0,0,1,0,0])
    elif word_label == 'restaurant' or 'restaurant' in word_label:
        return np.array([0,0,0,0,0,0,0,0,1,0])
    elif word_label == 'toystore' or 'toystore' in word_label:
        return np.array([0,0,0,0,0,0,0,0,0,1])

def Augmentation():
    IMAGE_SIZE = 80
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Each image is resized individually as different image may be of different size.
        for d in (os.listdir(TRAIN_DIR)):
            path = os.path.join(TRAIN_DIR, d)
            for k in (os.listdir(path)):
                if (k == 'toystore'):
                    pa = os.path.join(path, k)
                    for im in (os.listdir(pa)):
                        p = os.path.join(pa, im)
                        img = mpimg.imread(p)[:, :, :3]  # Do not read alpha channel.
                        resized_img = sess.run(tf_img, feed_dict={X: img})
                        resized_img = np.array(resized_img, dtype=np.float32)
                        NP = ''.join([im, 'N1.jpg'])
                        scipy.misc.imsave(NP, resized_img)
    return


def create_train_data():
    training_data = []
    for d in (os.listdir(TRAIN_DIR)):
      path = os.path.join(TRAIN_DIR, d)
      for k in (os.listdir(path)):
        pa = os.path.join(path, k)
        for img in (os.listdir(pa)):
          p = os.path.join(pa, img)
          img_data = cv2.imread(p, 0)
          img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
          training_data.append([np.array(img_data), create_label(k)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for d in (os.listdir(TEST_DIR)):
      path = os.path.join(TEST_DIR, d)
      for img in (os.listdir(path)):
          p = os.path.join(path, img)
          img_data = cv2.imread(p, 0)
          img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
          testing_data.append([np.array(img_data),create_label(img)])
    np.save('test_data.npy', testing_data)
    return testing_data

if (os.path.exists('train_data.npy')):
    train_data =np.load('train_data.npy')
    #train_data = create_train_data()
else: # If dataset is not created:
    train_data = create_train_data()
if (os.path.exists('test_data.npy')):
    test_data =np.load('test_data.npy')
else:
    test_data = create_test_data()


train = train_data
test = test_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 7, activation='relu')
pool2 = max_pool_2d(conv2, 7)

conv3 = conv_2d(pool2, 32, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 7, activation='relu')
pool4 = max_pool_2d(conv4, 7)

conv5 = conv_2d(pool4, 128, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 5, activation='softmax')
cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

if (os.path.exists('modell.tfl.meta')):
    model.load('./modell.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_test}, {'targets': y_test}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save('modell.tfl')

def CalcAcc():
    Right=0
    Wrong=0
    x=0
    for i in X_test:
        TestImage = i
        Lable=y_test[x]
        TestImage = cv2.resize(TestImage, (IMG_SIZE, IMG_SIZE))
        TestImage=TestImage.reshape(IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict([TestImage])[0]
        p=max(prediction)
        x+=1
        if(((p==prediction[0]) and (Lable[0]==1))):
            Right+=1
        elif(((p==prediction[1]) and (Lable[1]==1))):
            Right+=1
        elif (((p == prediction[2]) and (Lable[2]==1))):
            Right += 1
        elif (((p == prediction[3]) and (Lable[3]==1))):
            Right += 1
        elif (((p == prediction[4]) and (Lable[4]==1))):
            Right += 1
        elif (((p == prediction[5]) and (Lable[5]==1))):
            Right += 1
        elif (((p == prediction[6]) and (Lable[6]==1))):
            Right += 1
        elif (((p == prediction[7]) and (Lable[7]==1))):
            Right += 1
        elif (((p == prediction[8]) and (Lable[8]==1))):
            Right += 1
        elif (((p == prediction[9]) and (Lable[9]==1))):
            Right += 1
        else:
            Wrong+=1
    TotalAcc=(Right/240)*100
    return TotalAcc


def CSV():
    testing_data = []
    predictions=[]
    result=[]
    for img in (os.listdir(TEST_DIR)):
        pa = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(pa, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        img_data=np.array(img_data)
        prediction = model.predict([img_data])[0]
        p = max(prediction)
        testing_data.append(img)
        if (p == prediction[0]):
            out=1
        elif (p == prediction[1]):
            out=2
        elif (p == prediction[2]):
            out=3
        elif (p == prediction[3]):
            out=4
        elif (p == prediction[4]):
            out=5
        elif (p == prediction[5]):
            out=6
        elif (p == prediction[6]):
            out=7
        elif (p == prediction[7]):
            out=8
        elif (p == prediction[8]):
            out=9
        elif (p == prediction[9]):
            out=10
        predictions.append(out)
    for i in range(len(testing_data)):
        result.append([testing_data[i], predictions[i]])

    Csv = pd.DataFrame(result)
    Csv.to_csv("Result.csv")
   return

