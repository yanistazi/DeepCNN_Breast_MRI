# -*- coding: utf-8 -*-
import SimpleITK as sitk
from keras.preprocessing.image import ImageDataGenerator,  img_to_array
from scipy.misc import imresize
from skimage import exposure
import cv2
import os
import pandas as pd
import matplotlib
matplotlib.use('Tkagg')
from pylab import *

def normalize(image):
    return image / np.sqrt((np.sum(image ** 2)))


class Preprocessing(object):

    def __init__(self):
        self.path_train = '/Users/yanis/Desktop/train_set'
        self.path_val = '/Users/yanis/Desktop/sein_validation_set'
        self.size = 256
        self.percentile = (0.1, 99.9)
        self.save = '/Users/yanis/Desktop/ImageValidationDataChallengeJFR/'



    # Collect images

    def list_img_resized(self, train=True):
        list_path = []
        for dirName, subdirList, fileList in os.walk(self.path_train if train else self.path_val):
            for fileName in fileList:
                if '.nii.gz' in fileName.lower():
                    list_path.append(os.path.join(dirName, fileName))
        images_list = [''] * len(list_path)
        self.index = [int(filter(str.isdigit, lis)) for lis in list_path]
        for i, img in enumerate(list_path):
            image = sitk.ReadImage(img)
            images_list[i] = imresize(sitk.GetArrayFromImage(image), [self.size * sitk.GetArrayFromImage(image).shape[0]
                                                                    / sitk.GetArrayFromImage(image).shape[1], self.size])
        return images_list



    def get_result_df(self):
        df_val = pd.read_csv(self.path_val+'/validation_set.csv')
        df_val.drop(['Bénin'], axis=1, inplace=True)
        df_val = df_val.merge(pd.get_dummies(df_val['Type de lésion']), how='inner', on=df_val.index)
        df_val.drop(columns=['key_0', 'Type de lésion'], axis=1, inplace=True)
        df_val.rename(columns={'new_index': 'sein_index'}, inplace=True)
        df_train = pd.read_csv(self.path_train+'/train_set.csv')
        df_train.drop(['Bénin'], axis=1, inplace=True)
        df_train = df_train.merge(pd.get_dummies(df_train['Type de lésion']), how='inner', on=df_train.index)
        df_train.drop(columns=['key_0', 'Type de lésion'], axis=1, inplace=True)
        df_train = df_train.drop(df_train.index[166])
        df_train.reset_index(inplace=True)
        df_train.rename(columns={'Unnamed: 0': 'sein_index'}, inplace=True)
        result = pd.concat([df_train, df_val], sort=False)
        result.reset_index(inplace=True)
        result.fillna(0, inplace=True)
        result.drop(columns=['level_0', 'index'], inplace=True)
        return result


    def color_enhancement_imadjust(self):
        better_contrast_imadjust = [''] * len(self.list_img_resized())
        res = [''] * len(self.list_img_resized())
        for i, img in enumerate(self.list_img_resized()):
            v_min, v_max = np.percentile(img, self.percentile)
            better_contrast_imadjust[i] = exposure.rescale_intensity(img, in_range=(v_min, v_max))
            res[i] = np.hstack((better_contrast_imadjust[i], img))
        return better_contrast_imadjust


    def color_enhancement_adapthisteq(self, im_adjust=False):
        use = self.list_img_resized() if im_adjust is False else self.color_enhancement_imadjust()
        better_contrast_adapthisteq = [''] * len(use)
        res = [''] * len(use)
        for i, img in enumerate(use):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            better_contrast_adapthisteq[i] = clahe.apply(img)
            res[i] = np.hstack((better_contrast_adapthisteq[i], img))  # stacking images side-by-side
        return better_contrast_adapthisteq


    def reduce_noise(self):
        res = [''] * len(self.color_enhancement_adapthisteq(True))
        for i, img in enumerate(self.color_enhancement_adapthisteq(True)):
            res[i] = cv2.fastNlMeansDenoising(img, dst=None, h=5,
                                              templateWindowSize=7, searchWindowSize=21)
        return res


    def augmentator_shift_rotation(self, save_=False, img_to_create_per_sample=5,
                                   rotation_range=40, shift=0.2, range_=0.2, horizontal_flip=True):
        img = self.reduce_noise()
        datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=shift,
            height_shift_range=shift,
            shear_range=range_,
            zoom_range=range_,
            horizontal_flip=horizontal_flip,
            fill_mode='constant',
        )

        for j, im in enumerate(img):
            i = 0
            x = img_to_array(im)
            x = x.reshape((1,) + x.shape)
            for batch in datagen.flow(x, batch_size=1):
                if i < img_to_create_per_sample:
                    try:
                        batch = batch[0].reshape(self.size, self.size)
                        plt.imshow(batch, cmap=plt.cm.bone)
                        if save_:
                            plt.savefig(self.save + 'image_'+str(self.index[j])+'_augmentation_' + str(i) + '.jpg')
                        else:
                            plt.show()
                    except:
                        print self.index[j]
                    i += 1
                else:
                    break
        return


#get the output list
a=Preprocessing()
df_result = a.get_result_df()
list_train = a.list_img_resized()
list_train = sorted([str(li) for li in a.index if (li != 470)])
def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]

list_train = duplicate(list_train,5)
new_list = []
for l in list_train:
    new_list.append([l,df_result[df_result.sein_index=='sein_'+l].values.tolist()[0][1:]])
print len(new_list)
print new_list

list_val = a.list_img_resized(False)
list_val = sorted([str(li) for li in a.index ])
list_val = duplicate(list_val,5)
new_list_val = []
for l in list_val:
    new_list_val.append([l,df_result[df_result.sein_index=='validation_sein_'+l].values.tolist()[0][1:]])
print len(new_list_val)
print new_list_val
list_output=new_list + new_list_val
label=[l[1] for l in list_output]
import pickle
with open("labels.txt", "wb") as fp:   #Pickling
    pickle.dump(label, fp)
