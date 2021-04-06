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

img_width, img_height = 144, 144

def load_and_process( filename, row, model,  column_list ):
    img = cv2.imread( filename)
    img = cv2.resize( img, (img_height, img_width  ) )/255.0
    tensor_image = np.expand_dims( img, axis=0 )
    results = model.predict( tensor_image )
    results = results[0]

    lhs =  math.asin( np.clip( ( row[column_list[0]])*1.001, -1, 1 ) ) * 180/math.pi
    lhc =  math.acos( np.clip( ( row[column_list[1]])*1.001, -1, 1 ) ) * 180/math.pi
    lha  =  math.atan2( np.clip( ( row[column_list[0]])*1.001, -1, 1 ), np.clip( ( row[column_list[1]])*1.001, -1, 1 )   )* 180/math.pi

    lms =  math.asin( np.clip( ( row[column_list[2]])*1.001, -1, 1 ) ) * 180/math.pi
    lmc =  math.acos( np.clip( ( row[column_list[3]])*1.001, -1, 1 ) ) * 180/math.pi
    lma  =  math.atan2( np.clip( ( row[column_list[2]])*1.001, -1, 1 ), np.clip( ( row[column_list[3]])*1.001, -1, 1 )   )* 180/math.pi

    chs =  math.asin( np.clip( ( row[column_list[4]])*1.001, -1, 1 ) ) * 180/math.pi
    chc =  math.acos( np.clip( ( row[column_list[5]])*1.001, -1, 1 ) ) * 180/math.pi
    cha  =  math.atan2( np.clip( ( row[column_list[4]])*1.001, -1, 1 ), np.clip( ( row[column_list[5]])*1.001, -1, 1 )   )* 180/math.pi


    lh = lha/30
    lh =  round(lh)
    if lh < 0:
        lh = lh + 12

    lm = lma/6
    lm =  round(lm)
    if lm < 0:
        lm= lm + 60

    lch = cha
    lch = lch/30
    lch =  round(lch)
    if lch < 0:
        lch= lch + 12

    rhs =  math.asin( np.clip( ( results[0])*1.001, -1, 1 ) ) * 180/math.pi
    rhc =  math.acos( np.clip( ( results[1])*1.001, -1, 1 ) ) * 180/math.pi
    ha  =  math.atan2( np.clip( ( results[0])*1.001, -1, 1 ), np.clip( ( results[1])*1.001, -1, 1 )  )* 180/math.pi

    rms =  math.asin( np.clip( ( results[2])*1.001, -1, 1 ) ) * 180/math.pi
    rmc =  math.acos( np.clip( ( results[3])*1.001, -1, 1 ) ) * 180/math.pi
    ma  =  math.atan2( np.clip( ( results[2])*1.001, -1, 1 ), np.clip( ( results[3])*1.001, -1, 1 )  )* 180/math.pi

    hs =  math.asin( np.clip( ( results[4])*1.001, -1, 1 ) ) * 180/math.pi
    hc =  math.acos( np.clip( ( results[5])*1.001, -1, 1 ) ) * 180/math.pi
    cha =  math.atan2( np.clip( ( results[4])*1.001, -1, 1 ), np.clip( ( results[5])*1.001, -1, 1 )  )* 180/math.pi

    pm = ma/6
    pm =  round( pm + 0.5 )
    if pm < 0:
        pm = pm + 60
    pm = pm%60

    # Edge case
    ph   = ha # - ma/360
    phm  = round( ph)/6
    phmv = phm

    if pm > 54:
        if math.floor(phmv)%5 > 4 and  phm <2:
            pm = 0
        elif math.floor(phmv)%5<2:
            pm=0
    elif pm < 6:
            phmv = phmv +3
    elif pm > 30:
        phmv = phmv - 1

    ph = math.floor(phmv/5)

    if ph < 0:
        ph = ph + 12

    if ph != lch:
        print( "Label:- ",  lch, ":", lm,  " -- NG ",  int(ph), ":", int(pm), " ", round(ha/6), " ", phmv, phmv%5, round(ha) )  #  , "     ", int(pch), ":", int(pm) , " <------")
        print( "Label:- ",  lch, ":", lm,  " -- NG ",  int(ph), ":", int(pm), " Diff ", (lch-int(ph)), ":", (lm-(int(pm) ) ) )  #  , "     ", int(pch), ":", int(pm) , " <------")

    elif  abs(pm - lm)>1:
        print( "Label:- ",  lch, ":", lm,  " -- NG ",  int(ph), ":", int(pm), " Diff ", (lch-int(ph)), ":", (lm-(int(pm) ) ) )  #  , "     ", int(pch), ":", int(pm) , " <------")

    else:
        print( "Label:- ",  lch, ":",  lm  ) #  , "     ", int(pch), ":", int(pm) , " <------")
        pass


def model_test( csv, model_file ):
    # Lonad CSV Fi4
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

    # Model preparation
    input_shape = ( img_height, img_width,  3)
    input_img = Input( shape=input_shape )

    # Save model
    model = tensorflow.keras.models.load_model(model_file)
    model.summary()

    for index, row in df.iterrows():
        #print( "Row:-", index, row )
        load_and_process( row['filename'] , row, model,  column_list )


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
    model_test( csv, model_file )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('python clock_test.py -c <csv> -m <model>')
        sys.exit(2)
    main(sys.argv[1:])
