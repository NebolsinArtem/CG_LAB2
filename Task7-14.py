import numpy as np
from PIL import Image
from PIL import ImageOps
import math
import random

def parsedotsobj(file):
    arr = []
    for s in open(file):
        sp = s.split()
        if(sp[0] == 'v'):
          x = sp[1]
          y = sp[2]
          z = sp[3]
          arr.append([x, y, z])
    return arr

def parsepolysobj(file):
    arr = []
    for s in open(file):
        sp = s.split()
        if(sp[0] == 'f'):
            point1 = sp[1].split('/')[0]
            point2 = sp[2].split('/')[0]
            point3 = sp[3].split('/')[0]
            arr.append([point1, point2, point3])
          
    return arr


def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 =( (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 =( (x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return (lambda0, lambda1, lambda2)


def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    i = (y1 - y2) * (z1 - z0) - ((z1 - z2) * (y1 - y0))
    j = (x1 - x2) * (z1 - z0) - ((z1 - z2) * (x1 - x0))
    k = (x1 - x2) * (y1 - y0) - ((y1 - y2) * (x1 - x0))
    return (i, j, k)


def drawtriangle(x0, y0, x1, y1, x2, y2, img_mat, color, z_buffer):
    xmin = math.floor(min(x0, x1, x2))
    xmax = math.ceil(max(x0, x1, x2))

    ymin = math.floor(min(y0, y1, y2))
    ymax = math.ceil(max(y0, y1, y2))


    if (xmin < 0):
        xmin = 0

    if (ymin < 0):
        ymin = 0

    if (xmax > 2000):
        xmax = img_mat.shape[1]

    if (ymax > 2000):
        ymax = img_mat.shape[0]

    
    for i in range(int(xmin), int(xmax)):
        for j in range(int(ymin), int(ymax)):
        
            barycentric_cords = barycentric(i, j, x0, y0, x1, y1, x2, y2)

            if barycentric_cords[0] >= 0 and barycentric_cords[1] >= 0 and barycentric_cords[2] >= 0:
                barycentric_z = barycentric_cords[0] * z0 + barycentric_cords[1] * z1 + barycentric_cords[2] * z2
                if(barycentric_z < z_buffer[j,i]):
                   img_mat[j, i] = color
                   z_buffer[j,i] = barycentric_z





img_mat = np.zeros((2000,2000,3), dtype=np.uint8)

arrdots = parsedotsobj("C:\model_1.obj")
arrindexes = parsepolysobj("C:\model_1.obj")
z_buffer = np.full((2000, 2000), np.inf, dtype=np.float32)

for indexes in arrindexes:
    x0 = (float(arrdots[int(indexes[0]) - 1][0]))
    y0 = (float(arrdots[int(indexes[0]) - 1][1]))
    z0 = (float(arrdots[int(indexes[0]) - 1][2]))
    x1 = (float(arrdots[int(indexes[1]) - 1][0]))
    y1 = (float(arrdots[int(indexes[1]) - 1][1]))
    z1 = (float(arrdots[int(indexes[1]) - 1][2]))
    x2 = (float(arrdots[int(indexes[2]) - 1][0]))
    y2 = (float(arrdots[int(indexes[2]) - 1][1]))
    z2 = (float(arrdots[int(indexes[2]) - 1][2]))

    normal_coords = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cos_angle = normal_coords[2] / (math.sqrt(normal_coords[0] ** 2 + normal_coords[1] ** 2 + normal_coords[2] ** 2))

    if(cos_angle < 0):
        drawtriangle(x0 * 10000 + 1000, y0 * 10000 + 1000, x1 * 10000 + 1000, y1 * 10000 + 1000, x2 * 10000 + 1000, y2 * 10000 + 1000, img_mat, (int(-255*cos_angle), 0, 0), z_buffer)


img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.show()
img.save('img.png')



