# importing libraries
import sys, getopt
import math
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
import cv2
import numpy as np

import matplotlib.pyplot as plt

img_width, img_height = 100, 100


def to_grayscale_then_rgb(image):
    image = tensorflow.image.rgb_to_grayscale(image)
    image = tensorflow.image.grayscale_to_rgb(image)
    print(image.shape)
    return image

def model_build( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )
    #scaled_10  = BatchNormalization() (scaled_10a)

    feature_2 = Conv2D(64, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    hr_feature = Conv2D(40, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'
    mn_feature = Conv2D(30, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'

    hr_features = MaxPooling2D( (2,2))( hr_feature  )
    mn_features = MaxPooling2D( (2,2))( mn_feature  )
    # Minute hand features
    flat_mn  = Flatten()( mn_features )
    # Hour hand features
    flat_hr  = Flatten()( hr_features )
    # Hr hand features
    hrnd   = concatenate( [ flat_mn, flat_mn ] )

    mn11   = Dense(24, activation='relu')(flat_mn)
    mn12   = Dense(24, activation='relu')(flat_mn)

    hr11    = Dense(24, activation='relu')(flat_hr)
    hr12    = Dense(24, activation='relu')(flat_hr)

    hrnd11  = Dense(24, activation='relu')(hrnd)
    hrnd12  = Dense(24, activation='relu')(hrnd)

    mn21    = Dense(12, activation='relu')(mn11)
    mn22    = Dense(12, activation='relu')(mn12)

    hr21    = Dense(12, activation='relu')(hr11)
    hr22    = Dense(12, activation='relu')(hr12)

    hrnd21  = Dense(24, activation='relu')(hrnd11)
    hrnd22  = Dense(24, activation='relu')(hrnd12)
    # Rounded
    mn31    = Dense(1, activation='tanh')(mn21)
    mn32    = Dense(1, activation='tanh')(mn22)

    hr31    = Dense(1, activation='tanh')(hr21)
    hr32    = Dense(1, activation='tanh')(hr22)

    hrnd31    = Dense(1, activation='tanh')(hrnd21)
    hrnd32    = Dense(1, activation='tanh')(hrnd22)

    output = concatenate( [ hr31, hr32, mn31, mn32, hrnd31, hrnd32 ] )
    return  Model( [ input_img ], outputs = output )

def clock_train( csv, epochs=200, batch_size=2, saved_model=None ):
    df=pd.read_csv(csv, sep=',')
    print( df.head())
    nb_train_samples      = len(df.index)
    nb_validation_samples = len(df.index)

    print( "Columns ", df.columns)
    column_list = []
    index = 0
    for col in df.columns:
        print(col)
        if index != 0:
            column_list.append( col)
        index+=1
        print("Values ",  df[col].values)

        print(column_list, type(column_list) )

    model = model_build( img_height, img_width )

    model.compile(optimizer='adam', loss='mse', metrics =['accuracy'])
    print( model.summary() )

    # Image preprocessing
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        horizontal_flip = False,
        brightness_range=[0.4,1.0],
        rotation_range=2,
        zoom_range=[0.9,1.1],
        height_shift_range=[-4,4],
        width_shift_range=[-4,4],
        preprocessing_function=tensorflow.image.rgb_to_grayscale# to_grayscale_then_rgb
        )

    test_datagen = ImageDataGenerator(rescale = 1. / 255,
        preprocessing_function=tensorflow.image.rgb_to_grayscale ) # to_grayscale_then_rgb)

    # Training dataset
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        color_mode='rgb',
        y_col= column_list,
        target_size =( img_height, img_width ),
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'raw' )

    # Validation dataset
    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        color_mode='rgb',
        y_col= column_list,
        target_size =( img_height, img_width ),
        batch_size =  batch_size,
        class_mode ='raw' )

    # Fit Model
    model.fit(train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs, validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)

    # Save model
    if saved_model is not None:
        tensorflow.saved_model.save( model, saved_model)


def main( argv ):
    """
    :param argv: Command line arguments
    """
    csv = None # CSV File
    model_file = None
    try:
        opts, args = getopt.getopt(argv,"hc:m:",["csv", "model"])
    except getopt.GetoptError:
        print('python clockReaderTrain.py -c <csv> -m <model>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python clockReaderTrain.py -c <csv> -m <model>')
            sys.exit()
        elif opt in ("-c", "--csv"):
            csv = arg
        elif opt in ("-m", "--model"):
            model_file = arg

    if csv is None or model_file is None:
        print('python clockReaderTrain.py -c <csv> -m <model>')
        exit(2)

    print(" Csv file ", csv, " Model ", model_file  )
    clock_train( csv, epochs=200, batch_size=2, saved_model=model_file )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('python clockReaderTrain.py -c <csv> -m <model>')
        sys.exit(2)
    main(sys.argv[1:])
