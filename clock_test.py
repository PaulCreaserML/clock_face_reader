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


def load_and_process( filename, input_shape, row, model,  column_list ):
    img  = cv2.imread( filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize( gray, ( input_shape[0], input_shape[1]  ) )/255.0
    tensor_image = np.expand_dims( gray, axis=0 )
    results = model.predict( tensor_image )
    results = results[0]

    lha  =  math.atan2( np.clip( ( row[column_list[0]]*1.01), -1, 1 ), np.clip( ( row[column_list[1]]*1.01), -1, 1 )   )* 180/math.pi
    lma  =  math.atan2( np.clip( ( row[column_list[2]]*1.01), -1, 1 ), np.clip( ( row[column_list[3]]*1.01), -1, 1 )   )* 180/math.pi

    if lha < 0:
        lha = lha + 360

    label_hours   = math.floor(lha/30)
    label_minutes = math.floor( ( lha-label_hours*30)*2 )

    lh = lha/30
    lh = round(lh)
    if lh < 0:
        lh = lh + 12

    lm = lma/6
    lm =  round(lm)
    if lm < 0:
        lm= lm + 60

    ha  =  math.atan2( np.clip( ( results[0]*1.01), -1, 1 ), np.clip( ( results[1]*1.01), -1, 1 )  )* 180/math.pi
    ma  =  math.atan2( np.clip( ( results[2]*1.01), -1, 1 ), np.clip( ( results[3]*1.01), -1, 1 )  )* 180/math.pi


    if ha < 0:
        ha = ha + 360

    pred_hours   = math.floor(ha/30)
    pred_minutes = math.floor( ( ha-pred_hours*30)*2 )

    if ma < 0:
        ma = ma +360

    if ha < 0:
        ha = ha +360

    pm = ma/6
    pm =  round( pm + 0.5 )
    if pm < 0:
        pm = pm + 60
    pm = pm%60

    pred_minutes_approx = math.floor(ma/6)

    if ( pred_minutes_approx > 56 ) and ( pred_minutes > 56):
        pred_minutes = pred_minutes_approx
    elif ( pred_minutes_approx < 4) and ( pred_minutes < 4 ):
        pred_minutes = pred_minutes_approx
    elif ( pred_minutes_approx > 3 and pred_minutes_approx < 57) and ( pred_minutes > 3 and pred_minutes < 57):
        pred_minutes = pred_minutes_approx
    elif ( pred_minutes > 0 and pred_minutes <  30 ) and pred_minutes_approx > 50:
        pred_minutes = pred_minutes_approx

    print( label_hours, ":", label_minutes, " - ", pred_hours, ":", pred_minutes, "  ", pred_minutes_approx )


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

    # Save model
    model = tensorflow.keras.models.load_model(model_file)
    model.summary()

    # Model preparation
    input_shape = model.get_layer('input_1').input_shape[0][1:]

    for index, row in df.iterrows():
        load_and_process( row['filename'] , input_shape, row, model,  column_list )


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
