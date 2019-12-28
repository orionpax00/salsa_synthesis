import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

connection = [(0,1), (0,1),(0,5), (1,2), (0,9),(2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11),(11,12), (10,17),(10,13), (13,14), (14,15), (15,16),(10,17), (17,18), (18,19), (19,20)]


def generated_gif(config):
    data_gen = np.load("results/image_at_epoch_{:04d}.npy".format(config.PARAMETERS.EPOCHS))
    for j in range(101):
        fig = plt.figure()
        ax = Axes3D(fig)
        arr = []
        for i in connection:
            data_point = ([data_gen[j][0][i[0]][2], data_gen[j][0][i[1]][2]], [data_gen[j][0][i[0]][1], data_gen[j][0][i[1]][1]], [data_gen[j][0][i[0]][0], data_gen[j][0][i[1]][0]])
            arr.append(data_point)

        xm=[]
        ym=[]
        zm=[]

        for i in data_gen[j][0]:
            xm.append(i[0])
            ym.append(i[1])
            zm.append(i[2])

        plt.plot(zm,xm,ym, "r.")

        for i in arr:
            plt.plot(i[0],i[2],i[1]) ## plotting order should be Z,X,Y not X,Y,Z
        plt.savefig('results/predicted_imgs/image{:04d}.png'.format(j))
        plt.close()

def actual_gif(config):
    actual_file_loc = os.path.join(config.DATA.DATA_LOC, os.listdir(config.DATA.DATA_LOC)[1])
    data_gen = np.load(actual_file_loc)
    for j in range(101):
        fig = plt.figure()
        ax = Axes3D(fig)
        arr = []
        for i in connection:
            data_point = ([data_gen[j][i[0]][2], data_gen[j][i[1]][2]], [data_gen[j][i[0]][1], data_gen[j][i[1]][1]], [data_gen[j][i[0]][0], data_gen[j][i[1]][0]])
            arr.append(data_point)

        xm=[]
        ym=[]
        zm=[]

        for i in data_gen[j]:
            xm.append(i[0])
            ym.append(i[1])
            zm.append(i[2])

        plt.plot(zm,xm,ym, "r.")

        for i in arr:
            plt.plot(i[0],i[2],i[1]) ## plotting order should be Z,X,Y not X,Y,Z
        plt.savefig('results/actual_imgs/image{:04d}.png'.format(j))
        plt.close()



def evaluate(config):
    actual_gif_path = os.path.join(os.getcwd(),"results" ,"actual_imgs")
    predicted_gif_path = os.path.join(os.getcwd(), "results","predicted_imgs")
    save_loc = os.path.join(os.getcwd(),"results")
    if not os.path.exists(actual_gif_path):
        os.mkdir(actual_gif_path)
    if not os.path.exists(predicted_gif_path):
        os.mkdir(predicted_gif_path)

    actual_gif(config)
    imgs = []
    for image in os.listdir(actual_gif_path):
        img = Image.open(os.path.join(actual_gif_path, image))
        imgs.append(img)
    imgs[0].save(os.path.join(save_loc, 'actual.gif'),
               save_all=True, append_images=imgs[1:], optimize=False, duration=40, loop=0)


    generated_gif(config)
    imgs = []
    for image in os.listdir(predicted_gif_path):
        img = Image.open(os.path.join(predicted_gif_path, image))
        imgs.append(img)
    imgs[0].save(os.path.join(save_loc, 'generated.gif'),
               save_all=True, append_images=imgs[1:], optimize=False, duration=40, loop=0)