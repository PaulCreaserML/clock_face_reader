# importing libraries
import sys, getopt
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
import cv2
import numpy as np

import matplotlib.pyplot as plt

img_width, img_height = 100, 100

def load_and_process( filename, row, interpreter,  column_list ):

    img = cv2.imread( filename)
    img = cv2.resize( img, (img_height, img_width  ) )/255.0
    tensor_image = tf.convert_to_tensor(  np.expand_dims( img, axis=0 ), dtype=tf.float32)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, tensor_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index )
    results= predictions.squeeze()

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
        lh= lh + 12

    lm = lma/6
    lm =  round(lm)
    if lm < 0:
        lm= lm + 60

    ch = cha
    ch = ch/30
    ch =  round(ch)
    if ch < 0:
        ch= ch + 12

    rhs =  math.asin( np.clip( ( results[0])*1.001, -1, 1 ) ) * 180/math.pi
    rhc =  math.acos( np.clip( ( results[1])*1.001, -1, 1 ) ) * 180/math.pi
    rha  =  math.atan2( np.clip( ( results[0])*1.001, -1, 1 ), np.clip( ( results[1])*1.001, -1, 1 )  )* 180/math.pi

    rms =  math.asin( np.clip( ( results[2])*1.001, -1, 1 ) ) * 180/math.pi
    rmc =  math.acos( np.clip( ( results[3])*1.001, -1, 1 ) ) * 180/math.pi
    rma  =  math.atan2( np.clip( ( results[2])*1.001, -1, 1 ), np.clip( ( results[3])*1.001, -1, 1 )  )* 180/math.pi

    hs =  math.asin( np.clip( ( results[4])*1.001, -1, 1 ) ) * 180/math.pi
    hc =  math.acos( np.clip( ( results[5])*1.001, -1, 1 ) ) * 180/math.pi
    ha =  math.atan2( np.clip( ( results[4])*1.001, -1, 1 ), np.clip( ( results[5])*1.001, -1, 1 )  )* 180/math.pi

    rm = rma/6
    rm = round(rm)
    if rm < 0:
        rm= rm + 60

    rh = rha -rma/360
    rh = rh/30
    rh =  round(rh)
    if rh < 0:
        rh= rh + 12

    rch = ha
    rch = rch/30
    rch =  round(rch)
    if rch < 0:
        rch= rch + 12

    if ch!=rch or lm != rm:
        print( "NG ", int(ch), ":", int(lm), "|",  int(rch), ":", int(rm) , " <------")
    else:
        print( "OK ", int(ch), ":", int(lm), "|",  int(rch), ":", int(rm) )


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

    with open(model_file, 'rb') as fid:
        tflite_model = fid.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    #tflite_interpreter = tf.lite.Interpreter(model_path=model_file)

    for index, row in df.iterrows():
        #print( "Row:-", index, row )
        load_and_process( row['filename'] , row, interpreter, column_list )


def main( argv ):
    """
    :param argv: Command line arguments
    """
    csv = None # CSV File
    model_file = None
    try:
        opts, args = getopt.getopt(argv,"hc:l:",["csv", "lite"])
    except getopt.GetoptError:
        print('python clock_lite_test.py -c <csv> -l <lite>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python clock_lite_test.py -c <csv> -l <lite>')
            sys.exit()
        elif opt in ("-c", "--csv"):
            csv = arg
        elif opt in ("-l", "--lite"):
            model_file = arg

    if csv is None or model_file is None:
        print('python clock_lite_test.py -c <csv> -l <lite>')
        exit(2)

    print(" Csv file ", csv, " Model ", model_file  )
    model_test( csv, model_file )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('python clock_lite_test.py -c <csv> -l <lite>')
        sys.exit(2)
    main(sys.argv[1:])
