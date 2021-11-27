import random

import numpy as np
from numpy import newaxis
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

class DataGeneratorUnet(keras.utils.Sequence):
    'Generates data for Keras'
    'In order show clearly our applied augmentations we change default patch size to near real image size but we can ' \
    'change it to 128*128'

    def __init__(self, imgs_list, masks_list, patch_size=(280, 280), batch_size=32, shuffle=True):

        self.imgs_list = imgs_list
        self.masks_list = masks_list
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.number_of_samples = len(imgs_list)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.imgs_list) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, Y = self.__data_generation(batch_indexes)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))
        Y = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))

        for (jj, ii) in enumerate(batch_indexes):
            aux_img = np.load(self.imgs_list[ii])
            aux_mask = np.load(self.masks_list[ii])

            # Implement data augmentation function
            img_aug, mask_aug = self.__data_augmentation(aux_img, aux_mask)

            aux_img_patch, aux_mask_patch = self.__extract_patch(img_aug, mask_aug)

            X[jj, :, :] = aux_img_patch
            Y[jj, :, :] = aux_mask_patch

        return X, Y

    def __data_augmentation(self, img, mask):
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=random.randint(90, 270),
            width_shift_range=random.uniform(0, 0.2),
            brightness_range=[0.5, 1.5],
            height_shift_range=random.uniform(0, 0.2),
            shear_range=random.uniform(0, 0.2),
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=[0.8, 1.2],
            fill_mode='reflect',
        )
        'We can also use the following'
        # img_aug = datagen.apply_transform(image_for_augmentation, 5)

        transform_parameters = datagen.get_random_transform(img.shape)
        image_for_augmentation = img[:, :, newaxis]
        mask_for_augmentation = mask[:, :, newaxis]

        img_aug = datagen.apply_transform(image_for_augmentation, transform_parameters)
        mask_aug = datagen.apply_transform(mask_for_augmentation, transform_parameters)
        return img_aug, mask_aug

    def __extract_patch(self, img, mask):
        crop_idx = [None] * 2

        crop_idx[0] = np.random.randint(0, img.shape[0] - self.patch_size[0])
        crop_idx[1] = np.random.randint(0, img.shape[1] - self.patch_size[1])
        img_cropped = img[crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                      crop_idx[1]:crop_idx[1] + self.patch_size[1]]
        mask_cropped = mask[crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                       crop_idx[1]:crop_idx[1] + self.patch_size[1]]

        return img_cropped, mask_cropped
