# color convertor

import math

import cv2
import numpy as np
from PIL import Image

# D65/2°
# convert lab range [0,100][-128,128]^2 to [0,255]^3
def rgb_to_lab(rgb):
    var_R = ( rgb[0] / 255 ) # R from 0 to 255
    var_G = ( rgb[1] / 255 ) # G from 0 to 255
    var_B = ( rgb[2] / 255 ) # B from 0 to 255


    if var_R > 0.04045: 
        var_R = math.pow( ( var_R + 0.055 ) / 1.055, 2.4 )
    else:
        var_R = var_R / 12.92

    if var_G > 0.04045:
        var_G = math.pow( ( var_G + 0.055 ) / 1.055, 2.4 )
    else:
        var_G = var_G / 12.92

    if var_B > 0.04045:
        var_B = math.pow( ( var_B + 0.055 ) / 1.055, 2.4 )
    else:
        var_B = var_B / 12.92

    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100

    # Observer. = 2°, Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    var_X = X / 95.047
    var_Y = Y / 100
    var_Z = Z / 108.883

    if var_X > 0.008856:
        var_X = math.pow(var_X, 1/3 )
    else:
        var_X = ( 7.787 * var_X ) + ( 16 / 116 )

    if var_Y > 0.008856:
        var_Y = math.pow(var_Y, 1/3 )
    else:
        var_Y = ( 7.787 * var_Y ) + ( 16 / 116 )

    if var_Z > 0.008856:
        var_Z = math.pow(var_Z, 1/3 )
    else:
        var_Z = ( 7.787 * var_Z ) + ( 16 / 116 )

    L = round((( 116 * var_Y ) - 16 ) / 100 * 255)
    A = round(500 * ( var_X - var_Y ) + 128)
    B = round(200 * ( var_Y - var_Z ) + 128)

    return [L, A, B]

def lab_to_rgb(lab):
    var_Y = ( lab[0] / 255 * 100 + 16 ) / 116
    var_X = (lab[1] - 128) / 500 + var_Y
    var_Z = var_Y - (lab[2] - 128) / 200

    if var_Y > 0.206893034422:
        var_Y = math.pow(var_Y,3)
    else:
        var_Y = ( var_Y - 16 / 116 ) / 7.787
    if var_X > 0.206893034422:
        var_X = math.pow(var_X,3)
    else:
        var_X = ( var_X - 16 / 116 ) / 7.787
    if var_Z > 0.206893034422:
        var_Z = math.pow(var_Z,3)
    else:
        var_Z = ( var_Z - 16 / 116 ) / 7.787

    X = 95.047 * var_X
    Y = 100 * var_Y
    Z = 108.883 * var_Z

    var_X = X / 100
    var_Y = Y / 100
    var_Z = Z / 100

    var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
    var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570

    if var_R > 0.0031308:
        var_R = 1.055 * math.pow(var_R,1 / 2.4) - 0.055
    else:
        var_R = 12.92 * var_R
    if var_G > 0.0031308:
        var_G = 1.055 * math.pow(var_G,1 / 2.4) - 0.055
    else:
        var_G = 12.92 * var_G
    if var_B > 0.0031308:
        var_B = 1.055 * math.pow(var_B,1 / 2.4) - 0.055
    else:
        var_B = 12.92 * var_B

    R = round(var_R * 255)
    G = round(var_G * 255)
    B = round(var_B * 255)
    return [R, G, B]

def range0to255(color):
    for i in range(len(color)):
        color[i] = 0 if color[i] < 0 else color[i]
        color[i] = 255 if color[i] > 255 else color[i]
    return color

def rgb_to_hex(rgb):
    rgb = (rgb[0], rgb[1], rgb[2])
    return '#%02x%02x%02x' % rgb
