
from keras.preprocessing.image import ImageDataGenerator,  img_to_array
from scipy.misc import imresize
from skimage import exposure
import cv2
import os
import pydicom
import matplotlib
matplotlib.use('Tkagg')
from pylab import *


def normalize(image):
    return image / np.sqrt((np.sum(image ** 2)))


class Preprocessing(object):

    def __init__(self):
        self.path = '/Users/yanis/PycharmProjects/DataChallengeJFR/Inputs'
        self.size = 256
        self.percentile = (0.1, 99.9)
        self.save = '/Users/yanis/Desktop/Outputs Preprocessing/imageComparisonWithNormalization'
        self.save_bis = '/Users/yanis/Desktop/Outputs Preprocessing/Augmentation/'


    # Collect images

    def dicom_path(self):
        list_path = []
        for dirName, subdirList, fileList in os.walk(self.path):
            for fileName in fileList:
                if '.dcm' in fileName.lower():
                    list_path.append(os.path.join(dirName, fileName))
        return list_path

    def resize(self):

        width = self.size
        images_list = [''] * len(self.dicom_path())
        for i, img in enumerate(self.dicom_path()):
            image = np.array(pydicom.read_file(img).pixel_array)
            images_list[i] = imresize(image, [width*image.shape[0]/image.shape[1], width])
        return images_list

    def color_enhancement_imadjust(self):
        better_contrast_imadjust = [''] * len(self.resize())
        res = [''] * len(self.resize())
        for i, img in enumerate(self.resize()):
            v_min, v_max = np.percentile(img, self.percentile)
            better_contrast_imadjust[i] = exposure.rescale_intensity(img, in_range=(v_min, v_max))
            res[i] = np.hstack((better_contrast_imadjust[i], img))
        return better_contrast_imadjust

    def color_enhancement_adapthisteq(self, im_adjust=False):
        use = self.resize() if im_adjust is False else self.color_enhancement_imadjust()
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

    def show_comparison_images(self, save_=False):
        images = self.reduce_noise()
        for i in range(len(images)):
            edges = cv2.Canny(images[i], 100, 200)
            plt.subplot(131), plt.imshow(normalize(self.resize()[i]), cmap=plt.cm.bone)
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(normalize(images[i]), cmap=plt.cm.bone)
            plt.title('Preprocessed Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(133), plt.imshow(normalize(edges), cmap=plt.cm.bone)
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.savefig(self.save + str(i) + '.jpg') if save_ is True else plt.show()

    def augmentator_shift_rotation(self, save_=False, img_to_create_per_sample=20,
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
                    batch = batch[0].reshape(self.size, self.size)
                    plt.imshow(batch, cmap=plt.cm.bone)
                    if save_:
                        plt.savefig(self.save_bis + 'image_'+str(j)+'_augmentation_' + str(i) + '.jpg')
                    else:
                        plt.show()
                    i += 1
                else:
                    break

        return




obj = Preprocessing()

obj.augmentator_shift_rotation()
