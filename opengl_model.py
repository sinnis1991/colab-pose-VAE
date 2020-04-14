
# this function will turn a point in 3D into its position in 2D, ox oy F_x F_y are the intrinsicMatrix parameters
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2

import os
from math import pi,cos,sin
from read_stl import stl_model

def estimate_3D_to_2D(ox,oy,FocalLength_x,FocalLength_y,a,b,g,x_trans,z_trans,r,points):

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

  worldOrientation = rm.T
  worldLocation = pos*1000

  rotationMatrix = worldOrientation.T
  translationVector = -np.matmul(worldLocation,worldOrientation.T)

  a = np.matmul(points,rotationMatrix)+np.tile(translationVector,[np.size(points,0),1])

  u =  ox-FocalLength_x*a[:,0]/a[:,2]
  v =  oy-FocalLength_y*a[:,1]/a[:,2]

  results = np.array([u,v]).T

  return results

class gl_ob(object):
    
  def __init__(self, width=128, height=128, batch_size=64, path = './model1', points_path = './points1'):
    self.width = width
    self.height = height
    self.batch_size = batch_size


    self.path = os.path.join("./colab-pose-VAE",path)
    file = stl_model(self.path)
    self.tri = file.tri

    self.display_width = 640
    self.display_height = 480
    self.resize_window_side = 480
    self.initiate()

    self.spe_points = np.load(points_path)

    self.ox = 3.245938552764519e+02
    self.oy = 2.634932055129686e+02
    self.FocalLength_x = 5.938737905768464e+02
    self.FocalLength_y = 5.939301207515520e+02

  def cube(self):

      glBegin(GL_TRIANGLES)

      for Tri in self.tri:
          glColor3fv(Tri['colors'])
          glVertex3fv(
              (Tri['p0'][0], Tri['p0'][1], Tri['p0'][2]))
          glVertex3fv(
              (Tri['p1'][0], Tri['p1'][1], Tri['p1'][2]))
          glVertex3fv(
              (Tri['p2'][0], Tri['p2'][1], Tri['p2'][2]))

      glEnd()
      
  def draw_sence( self, alpha = 0, beta = 0, gama = 0, x_trans = 0, z_trans = 0, radius = 0.3, draw_cube = True, draw_chess = False, draw_window = False):

      r = radius

      a = alpha
      b = beta
      g = gama

      r_x = [[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]]
      r_y = [[cos(b),0,sin(b)],[0,1,0],[-sin(b),0,cos(b)]]
      r_z = [[cos(g),-sin(g),0],[sin(g),cos(g),0],[0,0,1]]

      bag = np.matmul (np.matmul (r_y, r_x),r_z)

      rm = bag

      zz = np.array([rm[0,2], rm[1,2], rm[2,2]])
      yy = np.array([rm[0,1], rm[1,1], rm[2,1]])
      xx = np.array([rm[0,0], rm[1,0], rm[2,0]])

      x = r*-zz[0]
      y = r*-zz[1]
      z = r*-zz[2]

      pos = np.array([x,y,z])+x_trans*xx+z_trans*yy
      obj = pos + zz

      gluLookAt(pos[0],pos[1],pos[2],obj[0],obj[1],obj[2],yy[0],yy[1],yy[2])

      if draw_cube:
          self.cube()
          
  def initiate(self):

      self.display = (self.display_width, self.display_height)
      
  def static_sence(self, a=0, b=0, g=0, x=0, z=0, r=0.3):
      
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
      glPushMatrix()
      glMatrixMode (GL_PROJECTION)
      glLoadIdentity()

      scale = 0.0001
      glFrustum(-324*scale,(640-324)*scale,-(480-263)*scale,263*scale,594*scale,20)

      glMatrixMode(GL_MODELVIEW)
      glClearDepth(1.0)
      glDepthFunc(GL_LESS)
      glEnable(GL_DEPTH_TEST)
      glEnable(GL_POINT_SMOOTH)
      glPolygonMode(GL_FRONT, GL_FILL)
      glPolygonMode(GL_BACK, GL_FILL)

      self.draw_sence(a,b,g,x,z,r)

      glPopMatrix()

      img_buf = glReadPixelsub(0, 0,self.display_width, self.display_height, GL_RGB, GL_UNSIGNED_BYTE)
      img = np.frombuffer(img_buf, np.uint8).reshape(self.display_height, self.display_width, 3)
      
      return(img)
    
  def dynamic_sence(self, A=[pi/3.,pi/2.], B=[pi,pi/2.*3.], G=[-pi/16.,pi/16.], X=[0.005,0.025], Z=[-0.025,0.005], R=[0.135,0.155], if_seed = False):

    tmp_arr_set = np.zeros((self.batch_size,self.display_height,self.display_width,3))

    window_side = self.resize_window_side
    margin_check_count = 0

    x1 = 160
    x2 = 640
    y1 = 0
    y2 = 480

    if if_seed==False:

      for i in range(self.batch_size):

        if_window_in = False
        tmp_arr = None

        while(if_window_in!=True):

          a = np.random.uniform(A[0], A[1])
          b = np.random.uniform(B[0], B[1])
          g = np.random.uniform(G[0], G[1])
          x = np.random.uniform(X[0], X[1])
          z = np.random.uniform(Z[0], Z[1])
          r = np.random.uniform(R[0], R[1])

          spe_points2D = estimate_3D_to_2D(self.ox,self.oy,self.FocalLength_x,self.FocalLength_y,\
            a,b,g,x,z,r,self.spe_points)
          
          if_window_in = True

          for k in range(np.size(spe_points2D,0)):
            u = spe_points2D[k,0]
            v = spe_points2D[k,1]

            if u<x1 or u>=x2 or v<y1 or v>=y2:
              if_window_in = False
              margin_check_count = margin_check_count +1  
              break

          if if_window_in==True:

            tmp_arr = self.static_sence(a,b,g,x,z,r)
            tmp_arr_set[i,:,:,:] = tmp_arr

    if if_seed==True:

      if_window_in = False
      tmp_arr = None
      
      seed = 2019
      np.random.seed(seed)
      seeds = np.random.uniform(0,1,[self.batch_size*100,6])
      index = 0
      seed_index = 0

      while(index < self.batch_size):
        if_window_in = False
        tmp_arr = None

        while(if_window_in!=True):

          a = A[0]+ (A[1]-A[0])*seeds[seed_index,0]
          b = B[0]+ (B[1]-B[0])*seeds[seed_index,1]
          g = G[0]+ (G[1]-G[0])*seeds[seed_index,2]
          x = X[0]+ (X[1]-X[0])*seeds[seed_index,3]
          z = Z[0]+ (Z[1]-Z[0])*seeds[seed_index,4]
          r = R[0]+ (R[1]-R[0])*seeds[seed_index,5]

          spe_points2D = estimate_3D_to_2D(self.ox,self.oy,self.FocalLength_x,self.FocalLength_y,\
          a,b,g,x,z,r,self.spe_points)

          if_window_in = True

          for k in range(np.size(spe_points2D,0)):
            u = spe_points2D[k,0]
            v = spe_points2D[k,1]

            if u<=x1 or u>=x2 or v<=y1 or v>=y2:
              if_window_in = False
              margin_check_count = margin_check_count +1
              break

          if if_window_in==True:

            tmp_arr = self.static_sence(a,b,g,x,z,r)
            tmp_arr_set[index] = tmp_arr
            index = index + 1

          seed_index = seed_index+1

    return tmp_arr_set
    
  def out_put_fast(self,window_side=480,if_write = False, A=[pi/3.,pi/2.], B=[pi,pi/2.*3.], G=[-pi/16.,pi/16.], X=[0.005,0.025], Z=[-0.025,0.005], R=[0.135,0.155], if_seed = False, option='M'):

    tmp_arr_set = self.dynamic_sence(A = A, B = B, G = G, X = X, Z = Z, R = R, if_seed = if_seed)
    result = np.zeros((self.batch_size,128,128))

    for i in range(self.batch_size):
      tmp_arr = tmp_arr_set[i]
      if if_write:
        cv2.imwrite('outfile_{}.png'.format(i), tmp_arr)
        
      if option == 'L':
        window_arr = tmp_arr[0:480, 0:480,  :]
      elif option == 'M':
        window_arr = tmp_arr[0:480, 80:560,  :]
      elif option == 'R':
        window_arr = tmp_arr[0:480, 160:640,  :]

      im_clip = cv2.resize(window_arr,(128*2,128*2))
      im_clip = im_clip.astype(np.uint8)
      canny_im = cv2.Canny(im_clip, 100, 200)
      kernel = np.ones((2, 2), np.uint8)
      dilation = cv2.dilate(canny_im, kernel, iterations=1)
      small = cv2.resize(dilation, (128,128))
      small = small.astype(np.uint8)

      small = np.array([small[-j] for j in range(len(small))])

      small = cv2.threshold(small,0,255,cv2.THRESH_BINARY)[1]
      
      if if_write:
        cv2.imwrite('outfile_clip_contour{}.png'.format(i), small)
      result[i] = small

    return result

  def read_pose_set(self, pose_set_path):
    self.pose = np.load(pose_set_path)
    self.pose_len = np.size(self.pose,0)
    self.pose_len = np.size(self.pose,0) 
    print ('pose_len:{}'.format(self.pose_len))
    a = self.pose_len%batch_size
    if a !=0:
      self.pose = np.concatenate((self.pose,self.pose[:self.batch_size-a,:]))
      self.pose_len = np.size(self.pose,0)
      print('pose_len_resized:{}'.format(self.pose_len))
    
    self.pose_batch_len = self.pose_len//64
    print('pose_batch_len:{}'.format(self.pose_batch_len))

  def dynamic_sence_poseset(self, pose_batch_index):

    tmp_arr_set = np.zeros((self.batch_size,self.display_height,self.display_width,3))

    for i in range(self.batch_size):
      p = self.pose[pose_batch_index*self.batch_size+i]
      tmp_arr = self.static_sence(p[0],p[1],p[2],p[3],p[4],p[5])
      tmp_arr_set[i] = tmp_arr 

    return tmp_arr_set

  def out_put_fast_poseset(self,window_side=480,if_write = False,pose_batch_index=0):

    tmp_arr_set = self.dynamic_sence_poseset(pose_batch_index)
    result = np.zeros((self.batch_size,128,128))

    for i in range(self.batch_size):
      tmp_arr = tmp_arr_set[i]
      if if_write:
        cv2.imwrite('outfile_{}.png'.format(i), tmp_arr)
        
      window_arr = tmp_arr[0:480, 160:640,  :]

      im_clip = cv2.resize(window_arr,(128*2,128*2))
      im_clip = im_clip.astype(np.uint8)
      canny_im = cv2.Canny(im_clip, 100, 200)
      kernel = np.ones((2, 2), np.uint8)
      dilation = cv2.dilate(canny_im, kernel, iterations=1)
      small = cv2.resize(dilation, (128,128))
      small = small.astype(np.uint8)

      small = np.array([small[-j] for j in range(len(small))])

      small = cv2.threshold(small,0,255,cv2.THRESH_BINARY)[1]
      
      if if_write:
        cv2.imwrite('outfile_clip_contour{}.png'.format(i), small)
      result[i] = small

    return result