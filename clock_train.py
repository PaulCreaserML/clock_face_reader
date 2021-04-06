# importing libraries
import sys, getopt
import math
import random
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
import numpy as np

import matplotlib.pyplot as plt

img_width, img_height = 120, 120

def test_preprocess(image):
    image = tensorflow.image.rgb_to_grayscale(image)
    image = tensorflow.image.grayscale_to_rgb(image)
    return image


def train_preprocess(image):
    #value = random.uniform(0, 1)
    image = tensorflow.image.rgb_to_grayscale(image)

    #noise = tensorflow.random.normal(shape=tensorflow.shape(image), mean=0.0, stddev=0.05, dtype=tensorflow.float32)
    #image = tensorflow.add(image, noise)

    #image = tensorflow.add( image, value/20.0 )

    #image = tensorflow.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    #if value > 0.5:
    #    image = 1.0 - image

    image = tensorflow.image.grayscale_to_rgb(image)
    return image

def model_build_couple( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(30, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )
    #scaled_10  = BatchNormalization() (scaled_10a)

    feature_2 = Conv2D(30, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    hr_feature = Conv2D(30, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'
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

    #
    mn21    = Dense(12, activation='relu')(mn11)
    mn22    = Dense(12, activation='relu')(mn12)

    hr21    = Dense(12, activation='relu')(hr11)
    hr22    = Dense(12, activation='relu')(hr12)

    hrnd21  = Dense(12, activation='relu')(hrnd11)
    hrnd22  = Dense(12, activation='relu')(hrnd12)

    #
    mn31    = Dense(1, activation='tanh')(mn21)
    mn32    = Dense(1, activation='tanh')(mn22)

    hr31    = Dense(1, activation='tanh')(hr21)
    hr32    = Dense(1, activation='tanh')(hr22)

    hrnd31   = concatenate( [ hr21, hrnd21] )
    hrnd32   = concatenate( [ hr22, hrnd22] )

    hrnd41    = Dense(1, activation='tanh')(hrnd31)
    hrnd42    = Dense(1, activation='tanh')(hrnd32)

    output = concatenate( [ hr31, hr32, mn31, mn32, hrnd41, hrnd42 ] )
    return  Model( [ input_img ], outputs = output )

def model_build_old( img_height, img_width, depth=3 ):
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


def model_build_simple( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )
    #scaled_10  = BatchNormalization() (scaled_10a)

    feature_2 = Conv2D(64, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(72, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'
    feature_2_scaled = MaxPooling2D( (2,2))( feature_3  )

    # Minute hand features
    flat_1  = Flatten()( feature_2_scaled )

    dense1  = Dense(48, activation='relu')(flat_1)

    #mn11    = Dense(24, activation='relu')(dense1)
    #mn12    = Dense(24, activation='relu')(dense1)

    #hr11    = Dense(24, activation='relu')(dense1)
    #hr12    = Dense(24, activation='relu')(dense1)

    #hrnd11  = Dense(24, activation='relu')(dense1)
    #hrnd12  = Dense(24, activation='relu')(dense1)

    #
    mn21    = Dense(12, activation='relu')(dense1)
    mn22    = Dense(12, activation='relu')(dense1)

    hr21    = Dense(12, activation='relu')(dense1)
    hr22    = Dense(12, activation='relu')(dense1)

    hrnd21  = Dense(24, activation='relu')(dense1)
    hrnd22  = Dense(24, activation='relu')(dense1)

    # Rounded
    mn31    = Dense(1, activation='tanh')(mn21)
    mn32    = Dense(1, activation='tanh')(mn22)

    hr31    = Dense(1, activation='tanh')(hr21)
    hr32    = Dense(1, activation='tanh')(hr22)

    hrnd31    = Dense(1, activation='tanh')(hrnd21)
    hrnd32    = Dense(1, activation='tanh')(hrnd22)

    output = concatenate( [ hr31, hr32, mn31, mn32, hrnd31, hrnd32 ] )
    return  Model( [ input_img ], outputs = output )


def model_build_simple_1( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2 = Conv2D(64, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(72, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'
    feature_2_scaled = MaxPooling2D( (2,2))( feature_3  )

    # Minute hand features
    flat_1  = Flatten()( feature_2_scaled )

    dense1  = Dense(48, activation='relu')(flat_1)

    #mn11    = Dense(24, activation='relu')(dense1)
    #mn12    = Dense(24, activation='relu')(dense1)

    #hr11    = Dense(24, activation='relu')(dense1)
    #hr12    = Dense(24, activation='relu')(dense1)

    #hrnd11  = Dense(24, activation='relu')(dense1)
    #hrnd12  = Dense(24, activation='relu')(dense1)

    #
    dense2   = Dense(48, activation='relu')(dense1)

    # Rounded
    mn31    = Dense(1, activation='tanh')(dense2)
    mn32    = Dense(1, activation='tanh')(dense2)

    hr31    = Dense(1, activation='tanh')(dense2)
    hr32    = Dense(1, activation='tanh')(dense2)

    hrnd31    = Dense(1, activation='tanh')(dense2)
    hrnd32    = Dense(1, activation='tanh')(dense2)

    output = concatenate( [ hr31, hr32, mn31, mn32, hrnd31, hrnd32 ] )
    return  Model( [ input_img ], outputs = output )

def model_build_simple_2( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2 = Conv2D(64, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(72, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'
    feature_2_scaled = MaxPooling2D( (2,2))( feature_3  )

    # Minute hand features
    flat_1  = Flatten()( feature_2_scaled )

    dense1  = Dense(48, activation='relu')(flat_1)

    #mn11    = Dense(24, activation='relu')(dense1)
    #mn12    = Dense(24, activation='relu')(dense1)

    #hr11    = Dense(24, activation='relu')(dense1)
    #hr12    = Dense(24, activation='relu')(dense1)

    #hrnd11  = Dense(24, activation='relu')(dense1)
    #hrnd12  = Dense(24, activation='relu')(dense1)

    #
    dense2   = Dense(48, activation='relu')(dense1)

    # Rounded
    output    = Dense(6, activation='tanh')(dense2)

    return  Model( [ input_img ], outputs = output )


def model_build_simple_4col( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )
    #scaled_10  = BatchNormalization() (scaled_10a)

    feature_2 = Conv2D(64, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(72, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'
    feature_2_scaled = MaxPooling2D( (2,2))( feature_3  )

    # Minute hand features
    flat_1  = Flatten()( feature_2_scaled )

    dense1  = Dense(48, activation='relu')(flat_1)

    mn21    = Dense(12, activation='relu')(dense1)
    mn22    = Dense(12, activation='relu')(dense1)

    hr21    = Dense(12, activation='relu')(dense1)
    hr22    = Dense(12, activation='relu')(dense1)

    # Rounded
    mn31    = Dense(1, activation='tanh')(mn21)
    mn32    = Dense(1, activation='tanh')(mn22)

    hr31    = Dense(1, activation='tanh')(hr21)
    hr32    = Dense(1, activation='tanh')(hr22)

    output = concatenate( [ hr31, hr32, mn31, mn32 ] )
    return  Model( [ input_img ], outputs = output )


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

    all_feature = Conv2D(70, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'

    all_features = MaxPooling2D( (2,2) )( all_feature  )
    # Minute hand features
    flat  = Flatten()( all_features )

    all11   = Dense(60, activation='relu')(flat)
    all12   = Dense(60, activation='relu')(flat)

    #
    mn21    = Dense(18, activation='relu')(all11)
    mn22    = Dense(18, activation='relu')(all12)

    hr21    = Dense(18, activation='relu')(all11)
    hr22    = Dense(18, activation='relu')(all12)

    hrnd21  = Dense(18, activation='relu')(all11)
    hrnd22  = Dense(18, activation='relu')(all12)


    mn31    = Dense(4, activation='relu')(mn21)
    mn32    = Dense(4, activation='relu')(mn22)

    hr31    = Dense(4, activation='relu')(hr21)
    hr32    = Dense(4, activation='relu')(hr22)

    hrnd31  = Dense(4, activation='relu')(hrnd21)
    hrnd32  = Dense(4, activation='relu')(hrnd22)

    # Last
    mn41    = Dense(1, activation='tanh')(mn31)
    mn42    = Dense(1, activation='tanh')(mn32)

    hr41    = Dense(1, activation='tanh')(hr31)
    hr42    = Dense(1, activation='tanh')(hr32)

    hrnd41    = Dense(1, activation='tanh')(hrnd31)
    hrnd42    = Dense(1, activation='tanh')(hrnd32)

    output = concatenate( [ hr41, hr42, mn41, mn42, hrnd41, hrnd42 ] )
    return  Model( [ input_img ], outputs = output )



def model_build_col_4( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )
    #scaled_10  = BatchNormalization() (scaled_10a)

    feature_2 = Conv2D(64, (3,3), padding='same', activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    all_feature = Conv2D(70, (3,3), activation='relu')( feature_2_scaled ) # , padding='same'

    all_features = MaxPooling2D( (2,2) )( all_feature  )
    # Minute hand features
    flat  = Flatten()( all_features )

    all11   = Dense(60, activation='relu')(flat)
    all12   = Dense(60, activation='relu')(flat)

    #
    mn21    = Dense(18, activation='relu')(all11)
    mn22    = Dense(18, activation='relu')(all12)

    hr21    = Dense(18, activation='relu')(all11)
    hr22    = Dense(18, activation='relu')(all12)

    mn31    = Dense(4, activation='relu')(mn21)
    mn32    = Dense(4, activation='relu')(mn22)

    hr31    = Dense(4, activation='relu')(hr21)
    hr32    = Dense(4, activation='relu')(hr22)

    # Last
    mn41    = Dense(1, activation='tanh')(mn31)
    mn42    = Dense(1, activation='tanh')(mn32)

    hr41    = Dense(1, activation='tanh')(hr31)
    hr42    = Dense(1, activation='tanh')(hr32)

    output = concatenate( [ hr41, hr42, mn41, mn42 ] )
    return  Model( [ input_img ], outputs = output )



def clock_train( csv, epochs=200, batch_size=2, saved_model=None, checkpoint_dir=None ):
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

    # model = model_build_couple( img_height, img_width )
    # model = model_build_old( img_height, img_width )
    # model = model_build_simple_2( img_height, img_width )
    # model = model_build_simple_4col( img_height, img_width )
    model = model_build_col_4( img_height, img_width )

    model.compile(optimizer='adam', loss='mse', metrics =['accuracy'])
    print( model.summary() )

    # Image preprocessing
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        horizontal_flip = False,
        brightness_range=[0.8,1.0],
        rotation_range=3,
        zoom_range=[0.9, 1.02],
        height_shift_range=[-4, 4],
        width_shift_range=[-4, 4],
        preprocessing_function=train_preprocess) # tensorflow.image.rgb_to_grayscale# to_grayscale_then_rgb

    test_datagen = ImageDataGenerator(rescale = 1. / 255,
        preprocessing_function=test_preprocess) # tensorflow.image.rgb_to_grayscale ) # to_grayscale_then_rgb)

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
        print('python clockReaderTrain.py -c <csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python clockReaderTrain.py -c <csv> -m <model> -k <checkpoint>')
            sys.exit()
        elif opt in ("-c", "--csv"):
            csv = arg
        elif opt in ("-m", "--model"):
            model_file = arg
        elif opt in ("-k", "--checkpoint"):
            checkpoint_dir = arg

    if csv is None or model_file is None:
        print('python clockReaderTrain.py -c <csv> -m <model> -k <checkpoint>')
        exit(2)

    print(" Csv file ", csv, " Model ", model_file  )
    clock_train( csv, epochs=120, batch_size=6, saved_model=model_file, checkpoint_dir=checkpoint_dir )


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print('python clockReaderTrain.py -c <csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    main(sys.argv[1:])
