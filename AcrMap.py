import numpy as np
from skimage import io, exposure
from matplotlib import pyplot as plt
from IPython.display import  Image
import colorsys



def read_band(path, n):
    if n in range(1,12):
        print path
        img = io.imread("/Users/jingjingxu/PycharmProjects/ArcMap/"+str(path)+'/LC81200402014162LGN00_B' + str(n) + '.tif',plugin='matplotlib')

        img = img.astype(float)
        return img
    else:
        print ('Band number has to be in the range 1-11')


def color_image_show(img, title):
    fig = plt.figure(figsize=(10,10))
    fig.set_facecolor('white')
    plt.imshow(img/65535.0)
    plt.title(title)
    plt.show()

def mix_brand(path,band_1,band_2,band_3):
    b2 = read_band(path,band_1)
    b3 = read_band(path,band_2)
    b4 = read_band(path,band_3)
    img432 = np.dstack((b4,b3,b2))
    return img432


#img_zhouwei_qiege = read_band("zhouwei_qiege", 3)
img_shuiti_qiege = mix_brand("shuiti_qiege_1", 5, 4, 3)
img_zhouwei_qiege = mix_brand("zhouwei_qiege", 5, 4, 3)
print "img_shuiti_qiege is ", img_shuiti_qiege
#img_shuiti_qiege = mix_brand("shuiti_qiege_1", 5, 4, 3)

#color_image_show(img_shuiti_qiege,'4-3-2 image, data set file')
#fig = plt.figure(figsize=(10, 7))
#fig.set_facecolor('white')

def compare_histogram(color_set,img1,img2):

    img1 = np.rollaxis(img1, axis=-1)
    img2 = np.rollaxis(img2, axis=-1)

    for color, channel_1,channel_2 in zip(color_set, img1,img2):

        plt.figure(color)

        counts_1, centers_1 = exposure.histogram(channel_1)
        counts_2, centers_2 = exposure.histogram(channel_2)

        selector_1 = counts_1 < 30000
        selector_2 = counts_2 < 30000

        counts_1 = counts_1[selector_1]
        centers_1 = centers_1[selector_1]

        counts_2 = counts_2[selector_2]
        centers_2 = centers_2[selector_2]

        counts_1 = counts_1.astype(float) / counts_1.sum()
        counts_2 = counts_2.astype(float) / counts_2.sum()

        plt.plot(centers_1[1::], counts_1[1::],label='water')
        plt.plot(centers_2[1::], counts_2[1::],label='land')

        plt.xlim((0,30000))

        plt.legend()

    plt.show()
    pass

def show_histogram(color_set, img):

    for color, channel in zip(color_set, np.rollaxis(img, axis=-1)):
        counts, centers = exposure.histogram(channel)
        plt.plot(centers[1::], counts[1::], color=color)

    plt.show()

#show_histogram('rgb',img_shuiti_qiege)

compare_histogram('rgb',img_shuiti_qiege,img_zhouwei_qiege)

plt.show()

def RGB_HSI(i):
    a = np.asarray(i, float)

    R, G, B = a.T

    print R,G,B

    m = np.min(a,2).T
    M = np.max(a,2).T

    C = M-m #chroma
    Cmsk = C!=0

    # Hue
    H = np.zeros(R.shape, float)
    mask = (M==R)&Cmsk
    H[mask] = np.mod(60.0*(G-B)/C, 360.0)[mask]
    mask = (M==G)&Cmsk
    H[mask] = (60.0*(B-R)/C + 120.0)[mask]
    mask = (M==B)&Cmsk
    H[mask] = (60.0*(R-G)/C + 240.0)[mask]
    H *= 255.0
    H /= 360.0 # if you prefer, leave as 0-360, but don't convert to uint8

    # Value
    V = M

    # Saturation
    S = np.zeros(R.shape, int)
    S[Cmsk] = ((255.0*C)/V)[Cmsk]

    I = (R+G+B)/3.0

    R_I=((R+1)/np.power(I+1.0,2.0))

    Total_stack = np.dstack((H,S,I,R_I))
    print "R_I is", R_I


    return Total_stack

Total_stack_1=RGB_HSI(img_shuiti_qiege)
Total_stack_2=RGB_HSI(img_zhouwei_qiege)

compare_histogram('HSIR',Total_stack_1,Total_stack_2)

