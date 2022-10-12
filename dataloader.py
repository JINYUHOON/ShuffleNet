import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import math
import albumentations as A
import argparse


class Generator(keras.utils.Sequence):
    def __init__(self, batch_size, df, img_size, mode, labels):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode
        self.labels = labels
        
        if mode == 'train':
            self.df = df[(df['fold'] != 1) & (df['fold'] != 2)]
        elif mode == 'valid':
            self.df = df[df['fold'] == 1]
        else:
            self.df = df[df['fold'] == 2]

        self.on_epoch_end()
        
        self.transform = A.Compose([
                        A.Blur(p=0.3, blur_limit=(3, 15)),
                        # # A.CLAHE(p=0.3, clip_limit=(1, 6), tile_grid_size=(20, 20)),
                        # # A.Downscale(p=0.3, scale_min=0.25, scale_max=0.25, interpolation=0),
                        # A.ElasticTransform(p=0.5, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
                        # # A.HorizontalFlip(p=0.3),
                        A.MotionBlur(p=0.3, blur_limit=(3, 30)),
                        A.VerticalFlip(p=0.3),
                        A.RandomScale(scale_limit=(-0.5,0) ,p=0.3, interpolation=1),
                        A.ShiftScaleRotate(p=1,
                        shift_limit=(-0.3, 0.3),
                         scale_limit=(-0.3, 3),
                          rotate_limit=(-90, 90),
                           interpolation=0,
                            border_mode=0,
                             value=(0, 0, 0), mask_value=None)],
                        # # A.RandomBrightnessContrast(p=0.3),
                        keypoint_params ={'format':'xy'})
    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)
        

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx+1) * self.batch_size
        data = self.df.iloc[start:end]
        
        batch_x , batch_y = self.get_data(data)
        
        if len(batch_x) != len(batch_y):
            batch_x = batch_x[:len(batch_y)]
        batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
        return batch_x, batch_y
    
    
    def get_data(self, data):
        batch_y = []
        X = np.ndarray((self.batch_size, self.img_size,self.img_size,3))
        
        for number, i in enumerate(data.index):
            image = cv2.imread(data.loc[i,'image_path'])
            
            xs, ys = image.shape[1] , image.shape[0]
 
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size,self.img_size))
            temp_y = np.array(data.iloc[number,1:-1])

            for i in range(0,self.labels,2):
                temp_y[i] = temp_y[i] * 256/xs
                temp_y[i+1] = temp_y[i+1] * 256/ys
                
            # Augmentation On Train Mode     
            if self.mode == 'train':
                tk = []
                for i in range(0,self.labels,2):
                    tk.append((temp_y[i], temp_y[i+1]))
                try:
                    augmented = self.transform(image = img, keypoints = tk)

                    aimg = augmented['image'].astype('float32')
                    aug_y = np.array(augmented['keypoints']).flatten()
                    
                    if len(aug_y) == self.labels:
                        X[number][:][:][:] = aimg/255.
                        batch_y.append(aug_y)
                    else:
                        X[number][:][:][:] = img/255.
                        batch_y.append(temp_y)
                except:
                    X[number][:][:][:] = img/255.
                    batch_y.append(temp_y)
                    
            else:
                  X[number][:][:][:] = img/255.
                  batch_y.append(temp_y)  

        batch_y = np.array(batch_y)
        
        return X, batch_y
    
    
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
