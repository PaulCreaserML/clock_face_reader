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

img_width, img_height = 144, 144

def test_preprocess(image):
    image = tensorflow.image.rgb_to_grayscale(image)
    image = tensorflow.image.grayscale_to_rgb(image)
    return image


def train_preprocess(image):
    image = tensorflow.image.rgb_to_grayscale(image)
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


def model_build_col_4_old( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )

    # average = Average()( [ input_img[:,:,:,0], input_img[:,:,:,1], input_img[:,:,:,2] ])
    # average = tensorflow.expand_dims(average, -1)  #axis=-4)#.shape.as_list()
    # reduce   =  Conv2D(1, (1,1), use_bias=False,  activation='relu')( input_img )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(64, (3,3),  use_bias=False, activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2 = Conv2D(64, (4,4), use_bias=False, activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    all_feature = Conv2D(64, (4,4), use_bias=False, activation='relu')( feature_2_scaled ) # , padding='same'
    all_features = MaxPooling2D( (2,2) )( all_feature  )
    # Minute hand features
    flat  = Flatten()( all_features )

    all11   = Dense(16, activation='relu')(flat)
    all12   = Dense(16, activation='relu')(flat)

    #
    mn21    = Dense(8)(all11)
    mn21    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn21)

    mn22    = Dense(8)(all12)
    mn22    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn22)

    hr21    = Dense(8)(all11)
    hr21    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr21)

    hr22    = Dense(8)(all12)
    hr22    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr22)

    #
    mn31    = Dense(4, activation='relu')(mn21)
    mn31    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn31)

    mn32    = Dense(4, activation='relu')(mn22)
    mn32    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn32)

    hr31    = Dense(4, activation='relu')(hr21)
    hr31    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr31)

    hr32    = Dense(4)(hr22)
    hr32    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr32)

    # Last
    mn41    = Dense(1, activation='tanh')(mn31)

    mn42    = Dense(1, activation='tanh')(mn32)

    hr41    = Dense(1, activation='tanh')(hr31)

    hr42    = Dense(1, activation='tanh')(hr32)

    output = concatenate( [ hr41, hr42, mn41, mn42 ] )
    return  Model( [ input_img ], outputs = output )


def custom_loss( y_true, y_pred ):
    loss = tensorflow.square( y_true - y_pred )
    return loss


def layer_div( value  ):
    #return tensorflow.math.add( value[0],  tensorflow.math.divide( value[1],  tensorflow.constant(60.0 )) )
    return value/60.0 # tensorflow.math.divide( value[1],  tensorflow.constant(60.0 ))



def model_build_col_4( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )

    # average = Average()( [ input_img[:,:,:,0], input_img[:,:,:,1], input_img[:,:,:,2] ])
    # average = tensorflow.expand_dims(average, -1)  #axis=-4)#.shape.as_list()
    # reduce   =  Conv2D(1, (1,1), use_bias=False,  activation='relu')( input_img )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(32, (3,3),  use_bias=False, activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2 = Conv2D(64, (3,3), use_bias=False, activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(48, (3,3), use_bias=False, activation='relu')( feature_2_scaled ) # batch_10 )
    feature_4 = Conv2D(48, (3,3), use_bias=False, activation='relu', padding='same')( feature_3 ) # , padding='same'
    skip1 = Add()( [ feature_3, feature_4 ])
    skip1_scaled = MaxPooling2D( (2,2))( skip1 )

    feature_5 = Conv2D(48, (3,3), use_bias=False, activation='relu')( skip1_scaled ) # batch_10 )
    feature_6 = Conv2D(48, (3,3), use_bias=False, activation='relu', padding='same')( feature_5 ) # , padding='same'
    skip2 = Add()( [ feature_5, feature_6 ])
    skip2_scaled = MaxPooling2D( (2,2))( skip2 )

    # Minute hand features
    flat  = Flatten()( skip2_scaled )

    all11   = Dense(24)(flat)
    all11   = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(all11)

    #all12   = Dense(16)(flat)
    #all12   = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(all12)

    #
    mn21    = Dense(8)(all11)
    mn21    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn21)

    mn22    = Dense(8)(all11)
    mn22    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn22)

    hr21    = Dense(8)(all11)
    hr21    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr21)

    hr22    = Dense(8)(all11)
    hr22    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr22)

    #
    mn31    = Dense(4)(mn21)
    mn31    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn31)

    mn32    = Dense(4)(mn22)
    mn32    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn32)

    hr31    = Dense(4)(hr21)
    hr31    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr31)

    hr32    = Dense(4)(hr22)
    hr32    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr32)


    # Last
    mn41    = Dense(1, activation='sigmoid')(mn31)
    #mn41    = tensorflow.keras.layers.ReLU(1.0, negative_slope=0.01)(mn41)

    mn42    = Dense(1, activation='sigmoid')(mn32)
    #mn42    = tensorflow.keras.layers.ReLU(1.0, negative_slope=0.01)(mn42)

    mn1_scale = Lambda( layer_div)(mn41)
    comb1 = concatenate( [hr31, mn1_scale] )

    mn2_scale = Lambda( layer_div)(mn42)
    comb2 = concatenate( [hr32, mn2_scale] )

    hr41    = Dense(1, activation='sigmoid')(comb1)
    #hr41    = tensorflow.keras.layers.ReLU(1.0, negative_slope=0.01)(hr41)

    hr42    = Dense(1, activation='sigmoid')(comb2)
    #hr42    = tensorflow.keras.layers.ReLU(1.0, negative_slope=0.01)(hr42)

    output = concatenate( [ hr41, hr42, mn41, mn42 ] )
    return  Model( [ input_img ], outputs = output )


def model_build_col_1( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )

    # average = Average()( [ input_img[:,:,:,0], input_img[:,:,:,1], input_img[:,:,:,2] ])
    # average = tensorflow.expand_dims(average, -1)  #axis=-4)#.shape.as_list()
    # reduce   =  Conv2D(1, (1,1), use_bias=False,  activation='relu')( input_img )
    # Feature Extraction For Both Hands
    feature_1 = Conv2D(32, (3,3),  use_bias=False, activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2 = Conv2D(64, (3,3), use_bias=False, activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(48, (3,3), use_bias=False, activation='relu')( feature_2_scaled ) # batch_10 )
    feature_4 = Conv2D(48, (3,3), use_bias=False, activation='relu', padding='same')( feature_3 ) # , padding='same'
    skip1 = Add()( [ feature_3, feature_4 ])
    skip1_scaled = MaxPooling2D( (2,2))( skip1 )

    feature_5 = Conv2D(48, (3,3), use_bias=False, activation='relu')( skip1_scaled ) # batch_10 )
    feature_6 = Conv2D(48, (3,3), use_bias=False, activation='relu', padding='same')( feature_5 ) # , padding='same'
    skip2 = Add()( [ feature_5, feature_6 ])
    skip2_scaled = MaxPooling2D( (2,2))( skip2 )

    # Minute hand features
    flat  = Flatten()( skip2_scaled )

    all    = Dense(24)(flat)
    all    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(all)

    #
    mn1    = Dense(8)(all)
    mn2    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn1)

    hr1    = Dense(8)(all)
    hr2    = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr1)

    #
    mn3     = Dense(4)(mn2)
    mn4     = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(mn3)

    hr3     = Dense(4)(hr2)
    hr4     = tensorflow.keras.layers.ReLU(6.0, negative_slope=0.01)(hr3)

    mn_div    = Lambda( layer_div )( mn4 )

    hr5 = Dense(1, activation='sigmoid')(hr4)
    mn5 = Dense(1, activation='sigmoid')(mn4)

    comb = concatenate( [ hr4, mn_div ])
    output = Dense(1, activation='sigmoid')(comb)

    output_all = concatenate( [ output, hr5, mn5 ] )

    return  Model( [ input_img ], outputs = output_all )





def model_build_col_6_nogood( img_height, img_width, depth=3 ):
    # Model build
    input_shape = ( img_height, img_width, depth )
    input_img = Input( shape=input_shape )

    feature_1 = Conv2D(32, (3,3),  use_bias=False, activation='relu')( input_img ) # padding='same',
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2 = Conv2D(64, (3,3), use_bias=False, activation='relu')( feature_1_scaled ) # batch_10 )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_3 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same')( feature_2_scaled ) # batch_10 )
    feature_4 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same')( feature_3 ) # , padding='same'
    skip1 = Add()( [ feature_3, feature_4 ])
    skip1_scaled = MaxPooling2D( (2,2))( skip1 )

    feature_5 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same')( skip1_scaled ) # batch_10 )
    feature_6 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same')( feature_5 ) # , padding='same'
    skip2 = Add()( [ feature_5, feature_6 ])
    skip2_scaled = MaxPooling2D( (2,2))( skip2 )

    # Minute hand features
    #flat1  = Flatten()( skip1_scaled )
    flat2  = Flatten()( skip2_scaled )

    #flat = concatenate( [ flat1, flat2] )

    all   = Dense(32, activation='relu')(flat2)

    #
    mn21    = Dense(12, activation='relu')(all)
    mn22    = Dense(12, activation='relu')(all)

    hr21    = Dense(12, activation='relu')(all)
    hr22    = Dense(12, activation='relu')(all)

    mn31    = Dense(6, activation='relu')(mn21)
    mn32    = Dense(6, activation='relu')(mn22)

    hr31    = Dense(6, activation='relu')(hr21)
    hr32    = Dense(6, activation='relu')(hr22)

    # Last
    mn41    = Dense(1, activation='tanh')(mn31)
    mn42    = Dense(1, activation='tanh')(mn32)

    hr41    = Dense(1, activation='tanh')(hr31)
    hr42    = Dense(1, activation='tanh')(hr32)

    # Hour Rounded
    rhr21    = concatenate( [ hr21, mn21 ])
    rhr22    = concatenate( [ hr22, mn22 ])

    rhr31    = Dense(8, activation='relu')(rhr21)
    rhr32    = Dense(8, activation='relu')(rhr22)

    rhr41    = Dense(1, activation='tanh')(rhr31)
    rhr42    = Dense(1, activation='tanh')(rhr32)

    output = concatenate( [ hr41, hr42, mn41, mn42, rhr41, rhr42 ] )
    return  Model( [ input_img ], outputs = output )

def model_build_col_6_c( img_height, img_width, depth=3 ):
    # Model build
    input_shape  = ( img_height, img_width, depth )
    input_img    = Input( shape=input_shape )

    feature_1    = Conv2D(64, (3,3),  use_bias=False, activation='relu')( input_img )
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2    = Conv2D(32, (3,3), use_bias=False, activation='relu')( feature_1_scaled )
    feature_2_scaled = MaxPooling2D( (2,2))( feature_2 )

    feature_1x1    = Conv2D(8, (1,1), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_3x3    = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_5x5    = Conv2D(32, (5,5), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    skip1          = concatenate( [ feature_1x1, feature_3x3, feature_5x5 ])
    skip1_scaled   = MaxPooling2D( (2,2))( skip1 )

    feature1_1x1   = Conv2D(8, (3,3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    feature1_3x3   = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    #feature1_5x5   = Conv2D(64, (5,5), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    skip2          = concatenate( [ feature_1x1, feature_3x3 ]) # , feature_5x5 ])
    skip2_scaled   = MaxPooling2D( (2,2))( skip2 )

    # Minute hand features
    flat2   = Flatten()( skip2_scaled )

    all     = Dense(64, activation='relu')(flat2)

    #
    mn21    = Dense(24, activation='relu')(all)
    mn22    = Dense(24, activation='relu')(all)

    hr21    = Dense(24, activation='relu')(all)
    hr22    = Dense(24, activation='relu')(all)

    # mn31    = Dense(8, activation='relu')(mn21)
    # mn32    = Dense(8, activation='relu')(mn22)

    # hr31    = Dense(8, activation='relu')(hr21)
    # hr32    = Dense(8, activation='relu')(hr22)

    # Last
    mn41    = Dense(1, activation='tanh')(mn21)
    mn42    = Dense(1, activation='tanh')(mn22)

    hr41    = Dense(1, activation='tanh')(hr21)
    hr42    = Dense(1, activation='tanh')(hr22)

    # Hour Rounded
    rhr21   = concatenate( [ hr21, mn21 ])
    rhr22   = concatenate( [ hr22, mn22 ])

    rhr31   = Dense(8, activation='relu')(rhr21)
    rhr32   = Dense(8, activation='relu')(rhr22)

    rhr41   = Dense(1, activation='tanh')(rhr31)
    rhr42   = Dense(1, activation='tanh')(rhr32)

    output  = concatenate( [ hr41, hr42, mn41, mn42, rhr41, rhr42 ] )

    return  Model( [ input_img ], outputs = output )


def model_build_col_6_bad( img_height, img_width, depth=3 ):
    # Model build
    input_shape  = ( img_height, img_width, depth )
    input_img    = Input( shape=input_shape )

    feature_1    = Conv2D(64, (5, 5), strides=(2, 2),  use_bias=False, activation='relu')( input_img )
    feature_1_scaled = MaxPooling2D( (2, 2) )( feature_1  )

    feature_2    = Conv2D(96, (3, 3), use_bias=False, activation='relu')( feature_1_scaled )
    feature_2_scaled = MaxPooling2D( (2, 2) )( feature_2 )

    feature_3x3    = Conv2D(64, (3, 3), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_5x5    = Conv2D(64, (5, 5), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    skip1          = concatenate( [ feature_3x3, feature_5x5 ])
    skip1_scaled   = MaxPooling2D( (2, 2) )( skip1 )

    feature1_3x3   = Conv2D(128, (3, 3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    #feature1_5x5   = Conv2D(64, (5,5), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    skip2_scaled   = MaxPooling2D( (2, 2) )( feature1_3x3 )

    # Minute hand features
    flat2   = Flatten()( skip2_scaled )

    all     = Dense(64, activation='relu')(flat2)
    all_drop= tensorflow.keras.layers.Dropout(.01)( all)
    #
    mn21_b  = Dense(60, activation='relu')(all_drop)
    mn21    = tensorflow.keras.layers.Dropout(.01)(mn21_b)

    mn22_b  = Dense(60, activation='relu')(all_drop)
    mn22    = tensorflow.keras.layers.Dropout(.01)(mn22_b)

    hr21_b  = Dense(60, activation='relu')(all_drop)
    hr21    = tensorflow.keras.layers.Dropout(.01)(hr21_b)

    hr22_b   = Dense(60, activation='relu')(all_drop)
    hr22     = tensorflow.keras.layers.Dropout(.01)(hr22_b)

    # Last
    mn41    = Dense(1, activation='tanh')(mn21)
    mn42    = Dense(1, activation='tanh')(mn22)
    hr41    = Dense(1, activation='tanh')(hr21)
    hr42    = Dense(1, activation='tanh')(hr22)

    # Hour Rounded
    rhr21   = concatenate( [ hr21, mn21 ])
    rhr22   = concatenate( [ hr22, mn22 ])

    rhr31   = Dense(8, activation='relu')(rhr21)
    rhr32   = Dense(8, activation='relu')(rhr22)
    rhr41   = Dense(1, activation='tanh')(rhr31)
    rhr42   = Dense(1, activation='tanh')(rhr32)

    output  = concatenate( [ hr41, hr42, mn41, mn42, rhr41, rhr42 ] )

    return  Model( [ input_img ], outputs = output )


def model_build_col_6_2( img_height, img_width, depth=3 ):
    # Model build
    input_shape      = ( img_height, img_width, depth )
    input_img        = Input( shape=input_shape )

    feature_1        = Conv2D(32, (3,3),  use_bias=False, activation='relu')( input_img )
    feature_1_scaled = MaxPooling2D( (2,2))( feature_1  )

    feature_2        = Conv2D(128, (3,3), use_bias=False, activation='relu')( feature_1_scaled )
    feature_2_1x1    = Conv2D(64, (1,1), use_bias=False, activation='relu', padding='same')( feature_2 )

    feature_2_scaled = MaxPooling2D( (2,2))( feature_2_1x1 )

    #
    feature_3_1x1    = Conv2D(16, (1,1), use_bias=False, activation='relu', padding='same')( feature_2_scaled )

    feature_3_3x3    = Conv2D(128, (3,3), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_3_3x3_1x1= Conv2D(64, (1,1), use_bias=False, activation='relu', padding='same')( feature_3_3x3 )

    feature_3_5x5    = Conv2D(128, (5,5), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_3_5x5_1x1= Conv2D(64, (1,1), use_bias=False, activation='relu', padding='same')( feature_3_5x5 )

    skip1            = concatenate( [ feature_3_1x1, feature_3_3x3_1x1, feature_3_5x5_1x1 ])
    skip1_scaled     = MaxPooling2D( (2,2))( skip1 )

    #
    feature_4_1x1     = Conv2D(8, (3,3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    feature_4_3x3     = Conv2D(64, (3,3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    #feature1_5x5    = Conv2D(64, (5,5), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    skip2            = concatenate( [ feature_4_1x1, feature_4_3x3 ]) # , feature_5x5 ])
    skip2_scaled     = MaxPooling2D( (2,2))( skip2 )

    # Minute hand features
    flat2   = Flatten()( skip2_scaled )

    all     = Dense(32, activation='relu')(flat2)

    #
    mn21    = Dense(12, activation='relu')(all)
    mn22    = Dense(12, activation='relu')(all)

    hr21    = Dense(12, activation='relu')(all)
    hr22    = Dense(12, activation='relu')(all)

    mn31    = Dense(6, activation='relu')(mn21)
    mn32    = Dense(6, activation='relu')(mn22)

    hr31    = Dense(6, activation='relu')(hr21)
    hr32    = Dense(6, activation='relu')(hr22)

    # Last
    mn41    = Dense(1, activation='tanh')(mn31)
    mn42    = Dense(1, activation='tanh')(mn32)

    hr41    = Dense(1, activation='tanh')(hr31)
    hr42    = Dense(1, activation='tanh')(hr32)

    # Hour Rounded
    rhr21   = concatenate( [ hr21, mn21 ])
    rhr22   = concatenate( [ hr22, mn22 ])

    rhr31   = Dense(16, activation='relu')(rhr21)
    rhr32   = Dense(16, activation='relu')(rhr22)

    rhr41   = Dense(1, activation='tanh')(rhr31)
    rhr42   = Dense(1, activation='tanh')(rhr32)

    output  = concatenate( [ hr41, hr42, mn41, mn42, rhr41, rhr42 ] )

    return  Model( [ input_img ], outputs = output )


def model_build_col_6( img_height, img_width, depth=3 ):
    # Model build
    input_shape  = ( img_height, img_width, depth )
    input_img    = Input( shape=input_shape )

    feature_1    = Conv2D(64, (5, 5), strides=(2, 2),  use_bias=False, activation='relu')( input_img )
    feature_1_scaled = MaxPooling2D( (2, 2) )( feature_1  )

    feature_2    = Conv2D(96, (3, 3), use_bias=False, activation='relu')( feature_1_scaled )
    feature_2_scaled = MaxPooling2D( (2, 2) )( feature_2 )

    feature_3x3    = Conv2D(64, (3, 3), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    feature_5x5    = Conv2D(64, (5, 5), use_bias=False, activation='relu', padding='same')( feature_2_scaled )
    skip1          = concatenate( [ feature_3x3, feature_5x5 ])
    skip1_scaled   = MaxPooling2D( (2, 2) )( skip1 )

    feature1_3x3   = Conv2D(64, (3, 3), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    #feature1_5x5   = Conv2D(64, (5,5), use_bias=False, activation='relu', padding='same')( skip1_scaled )
    #skip2_scaled   = MaxPooling2D( (2, 2) )( feature1_3x3 )

    # Minute hand features
    flat2   = Flatten()( feature1_3x3 ) # skip2_scaled )

    all     = Dense(64, activation='relu')(flat2)
    all_drop= tensorflow.keras.layers.Dropout(.01)( all)
    #
    mn21_b  = Dense(60, activation='relu')(all_drop)
    mn21    = tensorflow.keras.layers.Dropout(.01)(mn21_b)

    mn22_b  = Dense(60, activation='relu')(all_drop)
    mn22    = tensorflow.keras.layers.Dropout(.01)(mn22_b)

    hr21_b  = Dense(60, activation='relu')(all_drop)
    hr21    = tensorflow.keras.layers.Dropout(.01)(hr21_b)

    hr22_b   = Dense(60, activation='relu')(all_drop)
    hr22     = tensorflow.keras.layers.Dropout(.01)(hr22_b)

    # Last
    mn41    = Dense(1, activation='tanh')(mn21)
    mn42    = Dense(1, activation='tanh')(mn22)
    hr41    = Dense(1, activation='tanh')(hr21)
    hr42    = Dense(1, activation='tanh')(hr22)

    # Hour Rounded
    rhr21   = concatenate( [ hr21, mn21 ])
    rhr22   = concatenate( [ hr22, mn22 ])

    rhr31   = Dense(60, activation='relu')(rhr21)
    rhr32   = Dense(60, activation='relu')(rhr22)
    rhr41   = Dense(1, activation='tanh')(rhr31)
    rhr42   = Dense(1, activation='tanh')(rhr32)

    output  = concatenate( [ hr41, hr42, mn41, mn42] ) # , rhr41, rhr42 ] )

    return  Model( [ input_img ], outputs = output )


def model_build_col_red( img_height, img_width, depth=3 ):
    # Model build
    input_shape  = ( img_height, img_width, depth )
    input_img    = Input( shape=input_shape )

    feature_1    = Conv2D(64, (3, 3), strides=(2, 2),  use_bias=False, activation='relu')( input_img )
    feature_1_scaled = MaxPooling2D( (2, 2) )( feature_1  )

    feature_2    = Conv2D(96, (3, 3), use_bias=False, activation='relu')( feature_1_scaled )
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

    # model = model_build_couple( img_height, img_width )
    # model = model_build_old( img_height, img_width )
    # model = model_build_simple_2( img_height, img_width )
    # model = model_build_simple_4col( img_height, img_width )
    model =  model_build_col_red( img_height, img_width, depth=1 )

    # weights = [0.45, 0.45,  0.05, 0.05 ]
    model.compile(optimizer='adam', loss='mse', metrics =['accuracy']) # loss_weights=weights )
    # model.compile(optimizer='adam', loss='CosineSimilarity', metrics =['accuracy'])
    # model.compile(optimizer='adam', loss=custom_loss, metrics =['accuracy'])
    print( model.summary() )

    # Image preprocessing
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        #horizontal_flip = False,
        #brightness_range=[0.8,1.0],
        #zoom_range=[0.95, 1.05],
        # shear_range=0.01,
        #height_shift_range=[-10, 10],
        #width_shift_range=[-10, 10]#,
        #preprocessing_function=train_preprocess) # tensorflow.image.rgb_to_grayscale# to_grayscale_then_rgb
        )

    test_datagen = ImageDataGenerator(rescale = 1. / 255# ,
        # preprocessing_function=test_preprocess) # tensorflow.image.rgb_to_grayscale ) # to_grayscale_then_rgb
        )

    # Training dataset
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        color_mode='grayscale',#rgb',
        y_col= column_list,
        target_size =( img_height, img_width ),
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'raw' )

    # Validation dataset
    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        color_mode='grayscale',#'rgb',
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

    if csv is None or model_file is None or checkpoint_dir is None:
        print('python clockReaderTrain.py -c <csv> -m <model> -k <checkpoint>')
        exit(2)

    print(" Csv file ", csv, " Model ", model_file  )
    clock_train( csv, epochs=120, batch_size=16, saved_model=model_file, checkpoint_dir=checkpoint_dir )


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print('python clockReaderTrain.py -c <csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    main(sys.argv[1:])
