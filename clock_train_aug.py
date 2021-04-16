# importing libraries
import sys, getopt
import math
import random
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Average, Add, Multiply
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
import numpy as np

import matplotlib.pyplot as plt

clockface = None
diameter = 144

def model_build( img_height, img_width, depth=1 ):
    # Model build
    input_shape  = ( img_height, img_width, depth )
    input_img    = Input( shape=input_shape )

    feature_1    = Conv2D(64, (5, 5), strides=(2, 2),  use_bias=False, activation='relu')( input_img )
    feature_1_scaled = MaxPooling2D( (2, 2) )( feature_1  )

    feature_2    = Conv2D(96, (3, 3), use_bias=False, activation='relu')( feature_1_scaled)
    feature_2_scaled = MaxPooling2D( (2, 2) )( feature_2 )

    feature_3x3    = Conv2D(96, (3, 3), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_5x5    = Conv2D(12, (5, 5), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    skip1          = concatenate( [ feature_3x3, feature_5x5 ])
    skip1_scaled   = MaxPooling2D( (2, 2) )( skip1 )

    feature1_3x3   = Conv2D(72, (3, 3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    #feature1_5x5   = Conv2D(64, (5,5), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    skip2_scaled   = MaxPooling2D( (2, 2) )( feature1_3x3 )

    # Minute hand features feature1_3x3 ) #
    flat2   = Flatten()(  skip2_scaled )

    mn1_b  = Dense(64, activation='relu')(flat2)
    mn1    = tensorflow.keras.layers.Dropout(.01)(mn1_b)

    hr1_b   = Dense(64, activation='relu')(flat2)
    hr1     = tensorflow.keras.layers.Dropout(.01)(hr1_b)

    #
    mn2    = Dense(32, activation='relu')(mn1)
    hr2    = Dense(32, activation='relu')(hr1)

    # Last
    mn4    = Dense(2, activation='tanh')(mn2)
    hr4    = Dense(2, activation='tanh')(hr2)

    output  = concatenate( [ hr4, mn4] ) # , rhr41, rhr42 ] )

    return  Model( [ input_img ], outputs = output )

def set_clockFace( diameter ):
    image = tensorflow.keras.preprocessing.image.load_img("clockface1.png", color_mode= "grayscale")
    # print( image)
    image = tensorflow.keras.preprocessing.image.img_to_array(image)
    image = tensorflow.cast( image, dtype=tensorflow.uint8 )
    clockface = tensorflow.image.resize(image, [ diameter, diameter] )
    return clockface

def clockface_add(image):
    clockface = set_clockFace( 144 )
    image = tensorflow.math.add_n([clockface, image])
    return image


def clock_train( csv, epochs=200, batch_size=2, saved_model=None, checkpoint_dir=None ):
    df=pd.read_csv(csv, sep=',')
    print( df.head())
    nb_train_samples      = len(df.index)
    nb_validation_samples = len(df.index)

    column_list = []
    index = 0
    for col in df.columns:
        if index != 0:
            column_list.append( col)
        index+=1

    clockface = set_clockFace( diameter )
    model = model_build( diameter, diameter, depth=1 )

    model.compile(optimizer='adam', loss='mse', metrics =['accuracy'])
    print( model.summary() )

    # Image preprocessing
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        horizontal_flip = False,
        brightness_range=[0.8,1.2],
        zoom_range=[0.95, 1.05],
        # shear_range=0.01,
        height_shift_range=[-5, 5],
        width_shift_range=[-5, 5],
        preprocessing_function=clockface_add
        )

    test_datagen = ImageDataGenerator(rescale = 1. / 255
        )

    # Training dataset
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        color_mode='grayscale',
        y_col= column_list,
        target_size =( diameter, diameter ),
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'raw' )

    # Validation dataset
    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        color_mode='grayscale',
        y_col= column_list,
        target_size =( diameter, diameter ),
        batch_size =  batch_size,
        class_mode ='raw' )

    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # Fit Model
    model.fit(train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs, validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size,
        callbacks=[model_checkpoint_callback] )

    # Save model
    if saved_model is not None:
        tensorflow.saved_model.save( model, saved_model)


def main( argv ):
    """
    :param argv: Command line arguments
    """
    csv            = None # CSV File
    model_file     = None
    checkpoint_dir = None
    try:
        opts, args = getopt.getopt(argv,"hc:m:k:",["csv", "model", "checkpoint" ])
    except getopt.GetoptError:
        print('python clock_train.py -c <csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python clock_train.py -c <csv> -m <model> -k <checkpoint>')
            sys.exit()
        elif opt in ("-c", "--csv"):
            csv = arg
        elif opt in ("-m", "--model"):
            model_file = arg
        elif opt in ("-k", "--checkpoint"):
            checkpoint_dir = arg

    if csv is None or model_file is None or checkpoint_dir is None:
        print('python clock_train.py -c <csv> -m <model> -k <checkpoint>')
        exit(2)

    print(" Csv file ", csv, " Model ", model_file  )
    clock_train( csv, epochs=120, batch_size=16, saved_model=model_file, checkpoint_dir=checkpoint_dir )


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print('python clock_train.py -c <csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    main(sys.argv[1:])
