from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model



class ShuffleNet:
    def __init__(self, input_size, output_nums):
        self.input_size = input_size
        self.output_nums = output_nums
        
    def Shuffle_Net(self, *args, **kwargs):
        start_channels = 200
        groups = 2
        input = layers.Input((self.input_size, self.input_size, 3))
        
        x = layers.layers.Conv2D(24,kernel_size=3,strides = (2,2), padding = 'same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D (pool_size=(3,3), strides = 2, padding='same')(x)
        
        
        repetitions = [3,7,3]
        
        for i,repetition in enumerate(repetitions):
            channels = start_channels * (2**i)
            x  = self.shuffle_unit(x, groups, channels,strides = (2,2))
            
            for i in range(repetition):
                x = self.shuffle_unit(x, groups, channels,strides=(1,1))
                
                
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        
        output = layers.Dense(self.output_nums ,activation='linear')(x)
        model = Model(input, output)
    
        return model
    
    
    def shuffle_unit(self, x, groups, channels,strides):
        y = x
        
        out = layers.Conv2D(channels//4, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(x)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU()(out)
        
        out = self.channel_shuffle(out, groups)
        
        out = layers.DepthwiseConv2D(kernel_size = (3,3), strides = strides, padding = 'same')(out)
        out = layers.BatchNormalization()(out)
        
        if strides == (2,2):
            channels = channels - y.shape[-1]
        out = layers.Conv2D(channels, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(out)
        out = layers.BatchNormalization()(out)
        
        if strides ==(1,1):
            out = layers.Add()([out,y])
            
        if strides == (2,2):
            y = layers.AvgPool2D((3,3), strides = (2,2), padding = 'same')(y)
            out = layers.concatenate([out,y])
        out = layers.ReLU()(out)
        
        return out
    
    
    
    def channel_shuffle(self, x, groups):
        _, width, height, channels = x.get_shape().as_list()
        group_ch = channels // groups
        out = layers.Reshape([width, height, group_ch, groups])(x)
        out = layers.Permute([1, 2, 4, 3])(out)
        out = layers.Reshape([width, height, channels])(out)
        
        return out