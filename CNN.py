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
def create_label(image_name):
    word_label = image_name
    if word_label == 'Grenade' or 'Grenade' in word_label:
        return np.array([1,0,0,0,0])
    elif word_label == 'Machine Guns' or 'Machine Guns' in word_label:
        return np.array([0,1,0,0,0])
    elif word_label == 'Masked Face' or 'Masked Face' in word_label:
        return np.array([0,0,1,0,0])
    elif word_label == 'Pistol Hand Guns' or 'Pistol Hand Guns' in word_label:
        return np.array([0,0,0,1,0])
    elif word_label == 'RPG' or 'RPG' in word_label:
        return np.array([0,0,0,0,1])

def Augmentation():
    #IMAGE_SIZE = 100
    #tf.reset_default_graph()
    #X = tf.placeholder(tf.float32, (None, None, 3))
    #tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
                                    #tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        # Each image is resized individually as different image may be of different size.
    for d in (os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, d)
        if (d == 'airplane'):
            for im in (os.listdir(path)):
                p = os.path.join(path, im)
                img = mpimg.imread(p)[:, :, :3]  # Do not read alpha channel.
                #resized_img = sess.run(tf_img, feed_dict={X: img})
                #resized_img = np.array(resized_img, dtype=np.float32)
                flipped_img = np.fliplr(img)
                NP = ''.join([im, 'f.jpg'])
                scipy.misc.imsave(NP, flipped_img)
    return


def create_train_data():
    training_data = []
    for d in (os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, d)
        for img in (os.listdir(path)):
            p = os.path.join(path, img)
            img_data = cv2.imread(p, 0)
            try:
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                training_data.append([np.array(img_data), create_label(p)])
            except:
                pass
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in (os.listdir(TEST_DIR)):
        p = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(p, 0)
        try:
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img_data), create_label(img)])
        except:
            pass
    np.save('test_data.npy', testing_data)
    return testing_data

if (os.path.exists('train_data.npy')):
    train_data =np.load('train_data.npy')
    #train_data = create_test_data()
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

'''def CalcAcc():
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
'''
#Acc=CalcAcc()
#print("Total accuracy is:", Acc, "%")
#CSV()
#Augmentation()




'''def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)'''


#images = natural_sort(images)

'''im = cv2.imread('151.jpg')
# im = cv2.cvtColor(np.array(Image.open(images[0])), cv2.COLOR_BGR2GRAY)
# selective search
# im = cv2.resize(im, (newWidth, newHeight))
# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

# Switch to fast but low recall Selective Search method
# if (sys.argv[2] == 'f'):
ss.switchToSelectiveSearchFast()

# Switch to high recall but slow Selective Search method
# elif (sys.argv[2] == 'q'):
# ss.switchToSelectiveSearchQuality()
# if argument is neither f nor q print help message


# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# number of region proposals to show
numShowRects = 50
# increment to increase/decrease total number
# of reason proposals to be shown
increment = 50
NewRects=[]
while True:
    # create a copy of original image
    imOut = im.copy()

    # itereate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            NewRects.append(rect)
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    # show output
    cv2.imshow("Output", imOut)

    # record key press
    k = cv2.waitKey(0) & 0xFF

    # m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    # q is pressed
    elif k == 113:
        break
# close image show window
cv2.destroyAllWindows()
'''







#yabny da l code mlk4 da3wa bel ba2y
'''def tester(vid):

    video_cap = cv2.VideoCapture(vid)
    ret, im1 = video_cap.read()
    frame_rate = 0
    collectedFrames = 0
    while (True):
        # Capture frame-by-frame
        if frame_rate % 5 == 0:
            ret, im = video_cap.read()
            if ret:
                boxes = []
                preds = []
                maxs = []
            elif not ret:
                break
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(im)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        print('Total Number of Region Proposals: {}'.format(len(rects)))
        for i, rect in enumerate(rects):
            if i<=len(rects):
                x, y, w, h = rect
                if w >=270 and w<300 and h>=120 and h<200:
                    boxes.append(rect)
        print('Total Number of suspicious boxs: {}'.format(len(boxes)))
        imOut = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        for i, rect in enumerate(boxes):
            x, y, w, h = rect
            cropped = imOut[y:y+h , x:x+w]
            cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            cropped = cropped.reshape(IMG_SIZE, IMG_SIZE, 1)
            prediction = model.predict([cropped])[0]
            p = max(prediction)
            preds.append(p)
            if (p == prediction[0]):
                out = "Grenade"
            elif (p == prediction[1]):
                out = "Machine Guns"
            elif (p == prediction[2]):
                out = "Masked Face"
            elif (p == prediction[3]):
                out = "Pistol Hand Guns"
            elif (p == prediction[4]):
                out = "RPG"#m4 hnaaa da bm4y 3ala list gwa kol frame
            maxs.append(out)
        collectedFrames+=1
        if collectedFrames >= 150:
            break
        else:
            if len(boxes) !=0:
                ind=preds.index(max(preds))
                MC=maxs[ind]
                #MyRegions[ind]
                cv2.rectangle(imOut, (boxes[ind][0], boxes[ind][1]), (boxes[ind][0] + boxes[ind][2], boxes[ind][1] + boxes[ind][3]), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('Weapons Detection', imOut)

                import winsound
                winsound.PlaySound('s.wav', winsound.SND_FILENAME)
                #MC=max(((item, maxs.count(item)) for item in set(maxs)), key=lambda a: a[1])[0]
                print(MC)
                # wait for 'c' to close the application
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break
    video_cap.release()
    cv2.destroyAllWindows()
    frame_rate = frame_rate + 1


tester('w.mp4')
'''



def tester(vid):
    ret = True
    video_cap = cv2.VideoCapture(vid)
    video_cap.set(cv2.CAP_PROP_FPS, 1)
    while (video_cap.isOpened()):
        # Capture frame-by-frame
        ret, im = video_cap.read()
        boxes = []
        preds = []
        maxs = []
        if not ret:
            break
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(im)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        print('Total Number of Region Proposals: {}'.format(len(rects)))
        for i, rect in enumerate(rects):
            if i<=len(rects):
                x, y, w, h = rect
                if w >=270 and w<300 and h>=120 and h<200:
                    boxes.append(rect)
        print('Total Number of suspicious boxs: {}'.format(len(boxes)))
        imOut = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        for i, rect in enumerate(boxes):
            x, y, w, h = rect
            cropped = imOut[y:y+h , x:x+w]
            cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            cropped = cropped.reshape(IMG_SIZE, IMG_SIZE, 1)
            prediction = model.predict([cropped])[0]
            p = max(prediction)
            preds.append(p)
            if (p == prediction[0]):
                out = "Grenade"
            elif (p == prediction[1]):
                out = "Machine Guns"
            elif (p == prediction[2]):
                out = "Masked Face"
            elif (p == prediction[3]):
                out = "Pistol Hand Guns"
            elif (p == prediction[4]):
                out = "RPG"
            maxs.append(out)
        if len(boxes) !=0:
            ind=preds.index(max(preds))
            MC=maxs[ind]
            #MyRegions[ind]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imOut, MC, (boxes[ind][0], boxes[ind][1]) , font, .5, (0,0,0), 2, cv2.LINE_AA)
            cv2.rectangle(imOut, (boxes[ind][0], boxes[ind][1]), (boxes[ind][0] + boxes[ind][2], boxes[ind][1] + boxes[ind][3]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Weapons Detection', imOut)
            import winsound
            winsound.PlaySound('s.wav', winsound.SND_FILENAME)
            #MC=max(((item, maxs.count(item)) for item in set(maxs)), key=lambda a: a[1])[0]
            print(MC)
            # wait for 'c' to close the application
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()
    cv2.destroyAllWindows()
tester('w.mp4')