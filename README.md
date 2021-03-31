# clock_face_reader
Generates a model which can read the time from a analogue clock display

## Functionality

Takes a RGB image, resizes it to 100x100.

It generates 6 outputs, which is converted to 3 outputs using atan.

The outputs

Clock hour ( based on hour hand angle)
Clock hour ( an tttempt to get the actual hour, while compensating for the angle offset created by minutes)
Clock minute based on minute hand angle

## Robustness

Currently only works for pretrained images

## Future

Aim is to export an improved model and produce a demo for tenorflowjs ( Web page demo)
