import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from scipy.misc import imresize
from keras.layers import Convolution2D as Conv2D, MaxPooling2D, Input, Flatten, Dense
from keras.models import Sequential, Model, load_model
import os
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib
matplotlib.use('Tkagg')
from pylab import *
from keras import metrics,losses
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pickle

lr=0.00045
#momentum=0.9
decay=0.0
print('Adamax lr=', lr)
#print('momentum=', momentum)
print('decay=', decay)

def listings(path1="C:/Users/RLOCAL/Desktop/ImageValidationDataChallengeJFR/"
             ,path="C:/Users/RLOCAL/Desktop/ImageDataChallengeJFR/"):
    list_path_normal = []
    for dirName, subdirList, fileList in os.walk(path):
        for fileName in fileList:
            if '.jpg' in fileName.lower():
                list_path_normal.append(os.path.join(dirName, fileName))
    list_path_1 = []
    for dirName, subdirList, fileList in os.walk(path1):
        for fileName in fileList:
            if '.jpg' in fileName.lower():
                list_path_1.append(os.path.join(dirName, fileName))
    return list_path_normal, list_path_1


class Model_CNN(object):

    def __init__(self, model='vgg16'):
        self.name = model
        if self.name == 'vgg16':
            self.model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        elif self.name == 'resnet50':
            self.model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        elif self.name == 'inception_v3':
            self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        else:
            self.model = mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        self.list_path_normal = listings()[0]+listings()[1]  #train and after validation
        with open("C:/Users/RLOCAL/Desktop/DataChallenge/youtput.txt", "rb") as fp:  # Unpickling
            self.labels  = pickle.load(fp)
        self.mode = 'tf'

    def prepare_images(self):
        images = []
        for i, lis in enumerate(self.list_path_normal):
            images.append(imresize(np.array(load_img(lis)), [256, 256]))
            images[i] = img_to_array(images[i])
            #images[i] = np.expand_dims(images[i], axis=0)
            #images[i] = vgg16.preprocess_input(images[i].copy())
            if self.name == 'vgg16':
                images[i] = vgg16.preprocess_input(images[i].copy(), mode=self.mode)
            elif self.name == 'resnet50':
                images[i] = resnet50.preprocess_input(images[i].copy(), mode=self.mode)
            elif self.name == 'inception_v3':
                images[i] = inception_v3.preprocess_input(images[i].copy())
            else:
                images[i] = mobilenet.preprocess_input(images[i].copy())
        return images

    def model_preparation(self):
        #x = self.model.layers[5].output
        x=self.model.output
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
#        x = Dense(1024, activation='relu', name='fc2')(x)
#        x = Dense(1024, activation='relu', name='fc3')(x)
        x = Dense(512, activation='relu', name='fc4')(x)
#        x = Dense(256, activation='relu', name='fc5')(x)
        x = Dense(512, activation='relu', name='fc6')(x)
        predictions = Dense(18, activation='softmax', name='predictions')(x)
        model_final = Model(inputs=self.model.input, outputs=predictions)
        # Freeze the trainable layers except the last 4 layers
        for layer in self.model.layers :
            layer.trainable = False
        def auc(y_true, y_pred):
            try:
                score = tf.py_func(
                lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype(
                    'float32'),
                [y_true, y_pred]*18,
                'float32',
                stateful=False,
                name='sklearnAUC')
            except:
                pass
            return score

        def auc_roc(y_true, y_pred):
            # any tensorflow metric
            value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

            # find all variables created for this metric
            metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

            # Add metric variables to GLOBAL_VARIABLES collection.
            # They will be initialized for new session.
            for v in metric_vars:
                tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

            # force to update metric values
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
                return value

        model_final.compile(loss=["binary_crossentropy"], optimizer=optimizers.Adamax(lr=lr, decay=decay),metrics=[auc_roc])
        return model_final

    def training_evaluation(self, save = False):
        model_final=self.model_preparation()
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(self.prepare_images()), np.asarray(self.labels),
                                                            test_size=0.1)
        model_final.fit(x=X_train, y=y_train, batch_size=24, epochs=66, validation_split=0.2)
        if save:
            model_final.save('my_'+self.name+'.h5')
        score = model_final.evaluate(x=X_test, y=y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

a = Model_CNN()
a.training_evaluation(True)
#print len(a.prepare_images())
#plt.imshow(a.prepare_images()[0], cmap=plt.cm.bone)
#plt.show()
