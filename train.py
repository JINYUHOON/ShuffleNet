from gc import callbacks
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import argparse
import model
import dataloader
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', '--epoch', type=int, default=200)
    parser.add_argument('-image_size', '--image_size', type=int, default=256)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128)
    parser.add_argument('-data_path', '--data_path', type=str)
    parser.add_argument('-label_count', '--label_count', type=int)
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=0.001)

    return parser.parse_args()



def main(args):
    
    data = pd.read_csv(args.data_path)

    train_dataloader = dataloader.Generator(batch_size=args.batch_size,
                                            df = data, 
                                            img_size = args.image_size,
                                            mode = 'train')

    valid_dataloader = dataloader.Generator(batch_size=args.batch_size,
                                            df = data, 
                                            img_size = args.image_size,
                                            mode = 'valid')


    shufflenet = model.ShuffleNet(input_size = args.image_size,
                                output_nums = args.label_count)

    optim = keras.optimizers.Adam(args.learning_rate)

    shufflenet.compile(optimizers = optim, loss = 'mae')

    shufflenet.fit(train_dataloader, validation_data = valid_dataloader, epoch = args.epoch)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)