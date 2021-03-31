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

def load_and_process( filename, row, model, column_list ):
    img = cv2.imread( filename)
    img = cv2.resize( img, (img_height, img_width  ) )/255.0
    tensor_image = np.expand_dims( img, axis=0 )
    results = model.predict( tensor_image )
    results = results[0]

    lhs =  math.asin(  np.clip( ( row[column_list[0]])*1.001, -1, 1 ) ) * 180/math.pi
    lhc =  math.acos(  np.clip( ( row[column_list[1]])*1.001, -1, 1 ) ) * 180/math.pi
    lha =  math.atan2( np.clip( ( row[column_list[0]])*1.001, -1, 1 ), np.clip( ( row[column_list[1]] )*1.001, -1, 1 )  )* 180/math.pi

    lms =  math.asin(  np.clip( ( row[column_list[2]])*1.001, -1, 1 ) ) * 180/math.pi
    lmc =  math.acos(  np.clip( ( row[column_list[3]])*1.001, -1, 1 ) ) * 180/math.pi
    lma =  math.atan2( np.clip( ( row[column_list[2]])*1.001, -1, 1 ), np.clip( ( row[column_list[3]] )*1.001, -1, 1 )  )* 180/math.pi

    hs  =  math.asin(  np.clip( ( row[column_list[4]])*1.001, -1, 1 ) ) * 180/math.pi
    hc  =  math.acos(  np.clip( ( row[column_list[5]])*1.001, -1, 1 ) ) * 180/math.pi
    ha  =  math.atan2( np.clip( ( row[column_list[4]])*1.001, -1, 1 ), np.clip( ( row[column_list[5]] )*1.001, -1, 1 )  )* 180/math.pi


    h = lha/30
    h =  round(h)
    if h < 0:
        h= h + 12 #  24

    m = lma/6
    m =  round(m)
    if m < 0:
        m= m + 60

    ch = ha
    ch = ch/30
    ch =  round(ch)
    if ch < 0:
        ch= ch + 12 # 24

    print( filename )
    print( "Label ", int(lhs), int(lhc), int(lms), int(lmc), h, ch, m )

    # print("Label ", int(lhs), int(lhc), int(lha), int(lms), int(lmc), int(lma), int(hs), int(hc), int(ha) )

    rhs =  math.asin(  np.clip( ( results[0])*1.001, -1, 1 ) ) * 180/math.pi
    rhc =  math.acos(  np.clip( ( results[1])*1.001, -1, 1 ) ) * 180/math.pi
    rha =  math.atan2( np.clip( ( results[0])*1.001, -1, 1 ), np.clip( ( results[1] )*1.001, -1, 1 )  )* 180/math.pi

    rms =  math.asin(  np.clip( ( results[2])*1.001, -1, 1 ) ) * 180/math.pi
    rmc =  math.acos(  np.clip( ( results[3])*1.001, -1, 1 ) ) * 180/math.pi
    rma =  math.atan2( np.clip( ( results[2])*1.001, -1, 1 ), np.clip( ( results[3] )*1.001, -1, 1 )  )* 180/math.pi

    ms  =  math.asin(  np.clip( ( results[4])*1.001, -1, 1 ) ) * 180/math.pi
    mc  =  math.acos(  np.clip( ( results[5])*1.001, -1, 1 ) ) * 180/math.pi
    ma  =  math.atan2( np.clip( ( results[4])*1.001, -1, 1 ), np.clip( ( results[5] )*1.001, -1, 1 )  )* 180/math.pi

    m = ma/6
    m =  round(m)
    if m < 0:
        m= m + 60

    h = ha -ma/360
    h = h/30
    h =  round(h)
    if h < 0:
        h= h + 12 # 24

    ch = ha
    ch = ch/30
    ch =  round(ch)
    if ch < 0:
        ch= ch + 12 # 24

    print("Value", int(rhs), int(rhc), int(rms), int(rmc), h, ch, m )
    # print("Value ", int(rhs), int(rhc), int(rha), int(rms), int(rmc), int(rma) , int(ms), int(mc), int(ma) )


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

    # Final Check
    for index, row in df.iterrows():
        #print( "Row:-", index, row )
        load_and_process( row['filename'] , row, model, column_list  )


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
    clock_train( csv, epochs=100, batch_size=2, saved_model=model_file )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('python clockReaderTrain.py -c <csv> -m <model>')
        sys.exit(2)
    main(sys.argv[1:])
