import sys, getopt
import tensorflow as tf



def model_convert( model_file, lite_file ):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model( model_file )
    tflite_model = converter.convert()

    # Save the model.
    with open(lite_file, 'wb') as f:
        f.write(tflite_model)

    print("Complete!")

def main( argv ):
    """
    :param argv: Command line arguments
    """
    lite_file  = None
    model_file = None

    print(argv)
    try:
        opts, args = getopt.getopt( argv, "hm:l:", ["model", "lite"] )
    except getopt.GetoptError:
        print('python model_conv.py -m <model> -l <lite>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python model_conv.py -m <model> -l <lite>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model_file = arg
        elif opt in ("-l", "--lite"):
            lite_file = arg

    if lite_file is None or model_file is None:
        print('python model_conv.py -m <model> -l <lite>')
        exit(2)

    print(" Model ", model_file, " Lite ", lite_file  )
    model_convert( model_file, lite_file )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('python model_conv.py  -m <model> -l <lite>')
        sys.exit(2)
    main(sys.argv[1:])
