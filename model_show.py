
import pygame
from pygame.locals import *
import cv2
import scipy.misc
import math
from math import pi,cos,sin
# import glm


from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from read_stl import stl_model

file = stl_model()
tri = file.tri

def cube():
    
        glBegin(GL_TRIANGLES)
    
        for Tri in tri:
            glColor3fv(Tri['colors'])
            glVertex3fv(
                (Tri['p0'][0], Tri['p0'][1], Tri['p0'][2]))
            glVertex3fv(
                (Tri['p1'][0], Tri['p1'][1], Tri['p1'][2]))
            glVertex3fv(
                (Tri['p2'][0], Tri['p2'][1], Tri['p2'][2]))
    
        glEnd()

pygame.init()
display = (640,480)
window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)


W = .640
H = .480
near = .01
far = 2

L = -W/2.
R = W/2.
B = -H/2.
T = H/2.

N = near
F = far

ortho = np.zeros((4,4))
scale = np.sqrt(2)

ortho[0][0] = 2/(R-L)
ortho[0][3] = -(R+L)/(R-L)
ortho[1][1] =  2/(T-B) 
ortho[1][3] = -(T+B)/(T-B)
ortho[2][2] = -2/(F-N) 
ortho[2][3] = -(F+N)/(F-N)
ortho[3][3] =  1.0

glMatrixMode(GL_PROJECTION)
glLoadIdentity()

scale = 0.0001
glFrustum(-324*scale,(640-324)*scale,-(480-263)*scale,263*scale,594*scale,20)


glClearDepth(1.0)
glDepthFunc(GL_LESS)
glEnable(GL_DEPTH_TEST)
glEnable(GL_POINT_SMOOTH)
glPolygonMode(GL_FRONT, GL_FILL)
glPolygonMode(GL_BACK, GL_FILL)

def draw_cube(    eyeX=.0, eyeY=.0, eyeZ=.0,
                  centerX=.0, centerY=.0, centerZ=.0,
                  upX=.0, upY=1.0, upZ=.0,
                  transX=.0,transY=.0,transZ=.0):
        
        glPushMatrix()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
    	#1-4
        # a = 0.8052
        # b = 0.6065
        # g = -0.3147
        # r = 184.2338
        # x_trans = -23.2458
        # z_trans = 43.0278

        #5-8
        # a = 0.8251
        # b = -1.0248
        # g = -0.2846
        # r = 246.2197
        # x_trans = 56.9695
        # z_trans = 1.3111

        #9-12
        # a = 0.8137
        # b = 3.896
        # g = -0.304
        # r = 298.1812
        # x_trans = -3.6001
        # z_trans = -74.0879

        # #13-16
        # a = 0.8319
        # b = 2.1184
        # g = -0.3112
        # r = 252.9012
        # x_trans = -57.6401 
        # z_trans = -40.4607

        #17-20
        # a = 0.8134
        # b = 0.4004
        # g = -0.2981
        # r = 200.9494
        # x_trans = -13.1369
        # z_trans = 29.8525

        #21-24
        # a = 0.8125
        # b = -0.0942
        # g = -0.2881
        # r = 221.1005 
        # x_trans = 19.8832
        # z_trans = 19.1743

        #25-28
        # a = 0.803
        # b = -0.9089
        # g = -0.3028
        # r = 239.3215
        # x_trans = 41.0068
        # z_trans = 5.0604

        #29-32
        # a = 0.8233
        # b = -1.3424
        # g = -0.3055
        # r = 259.2587 
        # x_trans = 39.6502
        # z_trans = -15.8436

        #33-36
        # a = 0.3089
        # b = -0.8589
        # g = 4.0732
        # r = 197.8965
        # x_trans = -74.8187
        # z_trans = 13.4497

        #37-40
        a = 0.7914
        b = 0.4507
        g = -0.3299
        r = 186.5403
        x_trans = -30.655
        z_trans = 37.8485

        r = r/1000.
        x_trans = x_trans/1000.
        z_trans = z_trans/1000.

        r_x = [[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]]
        r_y = [[cos(b),0,sin(b)],[0,1,0],[-sin(b),0,cos(b)]]
        r_z = [[cos(g),-sin(g),0],[sin(g),cos(g),0],[0,0,1]]

        bag = np.matmul (np.matmul (r_y, r_x),r_z)

        rm = bag
     
        xx = np.array([rm[0,0], rm[1,0], rm[2,0]])
        yy = np.array([rm[0,1], rm[1,1], rm[2,1]])
        zz = np.array([rm[0,2], rm[1,2], rm[2,2]])

        x = r*-zz[0]
        y = r*-zz[1]
        z = r*-zz[2]
 
        pos = np.array([x,y,z])+x_trans*xx+z_trans*yy

        obj = pos + zz

        gluLookAt(pos[0],pos[1],pos[2],obj[0],obj[1],obj[2],yy[0],yy[1],yy[2])

        cube()
        glPopMatrix()
        pygame.display.flip()
        
        # Read the result
        string_image = pygame.image.tostring(window, 'RGB')
        temp_surf = pygame.image.fromstring(string_image, display, 'RGB')
        tmp_arr = pygame.surfarray.array3d(temp_surf)
        return (tmp_arr)

for i in range(10):

    im = draw_cube()

    if i == 1:
    	im2=np.zeros((480,640,3))
    	for m in range(480):
    		for n in range(640):
    			im2[m,n] = im[n,m]

        index =37

        cv2.imwrite('{}-{}.png'.format(index,index+3), im2)

        window_arr = im[0:480, 0:480, :]
        im_clip = cv2.resize(window_arr,(128*2,128*2))
        im_clip = im_clip.astype(np.uint8)
        canny_im = cv2.Canny(im_clip, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(canny_im, kernel, iterations=1)
        small = cv2.resize(dilation, (128,128))
        small = cv2.threshold(small,0,255,cv2.THRESH_BINARY)
        cv2.imwrite('L_{}-{}.png'.format(index,index+3), small[1].T)

        window_arr = im[160:640, 0:480, :]
        im_clip = cv2.resize(window_arr,(128*2,128*2))
        im_clip = im_clip.astype(np.uint8)
        canny_im = cv2.Canny(im_clip, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(canny_im, kernel, iterations=1)
        small = cv2.resize(dilation, (128,128))
        small = cv2.threshold(small,0,255,cv2.THRESH_BINARY)
        cv2.imwrite('R_{}-{}.png'.format(index,index+3), small[1].T)

        window_arr = im[80:560, 0:480, :]
        im_clip = cv2.resize(window_arr,(128*2,128*2))
        im_clip = im_clip.astype(np.uint8)
        canny_im = cv2.Canny(im_clip, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(canny_im, kernel, iterations=1)
        small = cv2.resize(dilation, (128,128))
        small = cv2.threshold(small,0,255,cv2.THRESH_BINARY)
        cv2.imwrite('M_{}-{}.png'.format(index,index+3), small[1].T)
        
