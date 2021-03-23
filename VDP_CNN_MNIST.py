import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import wandb
#os.environ["WANDB_API_KEY"] = "account_Key_code"
plt.ioff()
mnist = tf.keras.datasets.mnist
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def x_Sigma_w_x_T(x, W_Sigma):
  batch_sz = x.shape[0]
  xx_t = tf.reduce_sum(tf.multiply(x, x),axis=1, keepdims=True)               
  xx_t_e = tf.expand_dims(xx_t,axis=2)                                      
  return tf.multiply(xx_t_e, W_Sigma)

def w_t_Sigma_i_w(w_mu, in_Sigma):
  Sigma_1_1 = tf.matmul(tf.transpose(w_mu), in_Sigma)
  Sigma_1_2 = tf.matmul(Sigma_1_1, w_mu)
  return Sigma_1_2

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
  Sigma_3_1 = tf.linalg.trace(in_Sigma)
  Sigma_3_2 = tf.expand_dims(Sigma_3_1, axis=1)
  Sigma_3_3 = tf.expand_dims(Sigma_3_2, axis=1)
  return tf.multiply(Sigma_3_3, W_Sigma) 

def activation_Sigma(gradi, Sigma_in):
  grad1 = tf.expand_dims(gradi,axis=2)
  grad2 = tf.expand_dims(gradi,axis=1)
  return tf.multiply(Sigma_in, tf.matmul(grad1, grad2))

def activation_function_Sigma(gradi, Sigma_in):
   batch_size = gradi.shape[0]
   channels = gradi.shape[-1]  
   gradient_matrix = tf.reshape(gradi,[batch_size, -1, channels])# shape =[batch_size, image_size*image_size, channels]     
   grad1=tf.expand_dims(tf.transpose(gradient_matrix, [0, 2, 1]), 3)   # shape =[batch_size, channels, image_size*image_size, 1]  
   grad_square = tf.matmul(grad1, tf.transpose(grad1, [0, 1, 3,2]) )# shape =[batch_size, channels, image_size*image_size, image_size*image_size]   
   grad_square = tf.transpose(grad_square, [0, 2, 3, 1]) # shape =[ batch_size, image_size*image_size, image_size*image_size, channels]
   sigma_out = tf.multiply(Sigma_in, grad_square) 
   return sigma_out

def Hadamard_sigma(sigma1, sigma2, mu1, mu2):
  sigma_1 = tf.multiply(sigma1, sigma2)
  sigma_2 = tf.matmul(tf.matmul(tf.linalg.diag(mu1) ,   sigma2),   tf.linalg.diag(mu1))
  sigma_3 = tf.matmul(tf.matmul(tf.linalg.diag(mu2) ,   sigma1),   tf.linalg.diag(mu2))
  return sigma_1 + sigma_2 + sigma_3

def grad_sigmoid(mu_in):
  with tf.GradientTape() as g:
    g.watch(mu_in)
    out = tf.sigmoid(mu_in)
  gradi = g.gradient(out, mu_in) 
  return gradi

def grad_tanh(mu_in):
  with tf.GradientTape() as g:
    g.watch(mu_in)
    out = tf.tanh(mu_in)
  gradi = g.gradient(out, mu_in) 
  return gradi

def mu_muT(mu1, mu2):
  mu11 = tf.expand_dims(mu1,axis=2)
  mu22 = tf.expand_dims(mu2,axis=1)
  return tf.matmul(mu11, mu22)

def sigma_regularizer(x):      
    f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
    return tf.reduce_mean(-1. - tf.math.log(f_s) + f_s )

class VDP_first_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID"):
        super(VDP_first_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
    def build(self, input_shape):
        def sigma_regularizer_conv(x):      
            f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
            return (self.kernel_size* self.kernel_size*  input_shape[-1])*tf.reduce_mean(f_s-tf.math.log(f_s)-1.)
        ini_sigma = -2.2       
        tau = 1.# / (self.kernel_size* self.kernel_size*  input_shape[-1])
        self.w_mu = self.add_weight(name='w_mu', shape=(self.kernel_size, self.kernel_size,  input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    regularizer=tf.keras.regularizers.l2(tau),#l1_l2(l1=tau, l2=tau),                                    
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(name='w_sigma',
                                       shape=(self.kernel_num,),
                                       initializer=tf.constant_initializer(ini_sigma),#tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),
                                       regularizer=sigma_regularizer_conv,  
                                       trainable=True,
                                       )
    def call(self, mu_in):
        batch_size = mu_in.shape[0]
        num_channel = mu_in.shape[-1]
        mu_out = tf.nn.conv2d(mu_in, self.w_mu,  strides=[1,self.kernel_stride,self.kernel_stride,1], padding=self.padding, data_format='NHWC')
        x_train_patches = tf.image.extract_patches(mu_in, sizes=[1,self.kernel_size,self.kernel_size,1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)# shape=[batch_size, image_size, image_size, kernel_size*kernel_size*num_channel]
        x_train_matrix = tf.reshape(x_train_patches,[batch_size, -1, self.kernel_size*self.kernel_size*num_channel])# shape=[batch_size, image_size*image_size, patch_size*patch_size*num_channel]
        X_XTranspose = tf.matmul(x_train_matrix, tf.transpose(x_train_matrix,[0, 2, 1] ))# shape=[batch_size,image_size*image_size, image_size*image_size ]  in the tensor z
        X_XTranspose = tf.ones([1,1,1, self.kernel_num]) * tf.expand_dims(X_XTranspose, axis=-1)    
        Sigma_out = tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), X_XTranspose)#shape=[batch_size,image_size*image_size, image_size*image_size, kernel_num]      
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out		
		

class VDP_intermediate_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID"):
        super(VDP_intermediate_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
    def build(self, input_shape):
        def sigma_regularizer_conv(x):      
            f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
            return (self.kernel_size* self.kernel_size*  input_shape[-1])*tf.reduce_mean(f_s-tf.math.log(f_s)-1.)
        ini_sigma = -2.2
        #min_sigma = -4.5
        tau = 1. #/ (self.kernel_size* self.kernel_size*  input_shape[-1])
        self.w_mu = self.add_weight(name='w_mu', shape=(self.kernel_size, self.kernel_size,  input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    regularizer=tf.keras.regularizers.l2(tau),# l1_l2(l1=tau, l2=tau),                                  
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(name='w_sigma',
                                       shape=(self.kernel_num,),
                                       initializer=tf.constant_initializer(ini_sigma),#tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),
                                       regularizer=sigma_regularizer_conv,  
                                       trainable=True,
                                       )
    def call(self, mu_in, Sigma_in):
        batch_size = mu_in.shape[0]
        num_channel = mu_in.shape[-1] #shape=[batch_size, im_size, im_size, num_channel] 
        mu_out = tf.nn.conv2d(mu_in, self.w_mu,  strides=[1,self.kernel_stride,self.kernel_stride,1], padding=self.padding, data_format='NHWC')
        Sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2])
        diag_sigma = tf.linalg.diag_part(Sigma_in1)#shape=[batch_size, num_channel,im_size*im_size] 
        diag_sigma = tf.transpose(diag_sigma, [0, 2, 1]) #shape=[batch_size, im_size*im_size,num_channel] 
        diag_sigma = tf.reshape(diag_sigma, [batch_size, mu_in.shape[1], mu_in.shape[2], num_channel] ) #shape=[batch_size, im_size,im_size,num_channel]     
        diag_sigma_patches=tf.image.extract_patches(diag_sigma,sizes=[1,self.kernel_size,self.kernel_size,1],strides=[1,self.kernel_stride,self.kernel_stride,1],rates=[1,1,1,1],padding=self.padding)
        # shape=[batch_size, new_im_size, new_im_size, kernel_size*kernel_size*num_channel]
        diag_sigma_g = tf.reshape(diag_sigma_patches, [batch_size, -1, self.kernel_size*self.kernel_size*num_channel] )
        # shape=[batch_size, new_im_size*new_im_size,   self.kernel_size*self.kernel_size*num_channel ]
        mu_cov_square =  tf.reshape(tf.multiply(self.w_mu, self.w_mu ), [self.kernel_size*self.kernel_size*num_channel, self.kernel_num])
        # shape[ kernel_size*kernel_size*num_channel,   kernel_num]
        mu_wT_sigmags_mu_w1 = tf.matmul(diag_sigma_g, mu_cov_square)#shape=[batch_size, new_im_size*new_im_size , kernel_num   ]    
        mu_wT_sigmags_mu_w = tf.linalg.diag(tf.transpose(mu_wT_sigmags_mu_w1, [0, 2, 1])) #shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        trace = tf.math.reduce_sum(diag_sigma_g, 2, keepdims=True)# shape=[batch_size,  new_im_size* new_im_size, 1]
        trace = tf.ones([1, 1, self.kernel_num]) * trace #shape=[batch_size,  new_im_size*new_im_size, kernel_num]
        trace = tf.transpose(tf.multiply( tf.math.log(1. + tf.math.exp(self.w_sigma)), trace ),   [0, 2, 1])#shape=[batch_size, kernel_num, new_im_size*new_im_size]    
        trace1 = tf.linalg.diag(trace) #shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        mu_in_patches = tf.reshape(tf.image.extract_patches(mu_in ,sizes=[1,self.kernel_size ,self.kernel_size ,1],strides=[1,self.kernel_stride,self.kernel_stride,1],rates=[1,1,1,1],padding=self.padding), [batch_size, -1,self.kernel_size*self.kernel_size*num_channel ])
        # shape=[batch_size, new_im_size*new_im_size, self.kernel_size*self.kernel_size*num_channel]
        mu_gT_mu_g = tf.matmul(mu_in_patches, tf.transpose(mu_in_patches, [0, 2, 1]))# shape=[batch_size, new_im_size*new_im_size,new_im_size*new_im_size]
        mu_gT_mu_g1 = tf.ones([1, 1,1, self.kernel_num]) * tf.expand_dims(mu_gT_mu_g, axis=-1)
        # shape=[batch_size, new_im_size*new_im_size, new_im_size*new_im_size, kernel_num] 
        sigmaw_mu_gT_mu_g = tf.transpose(tf.multiply( tf.math.log(1. + tf.math.exp(self.w_sigma)), mu_gT_mu_g1 ), [0, 3 ,1 ,2])
        # shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        Sigma_out = trace1 + mu_wT_sigmags_mu_w + sigmaw_mu_gT_mu_g #shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        Sigma_out = tf.transpose(Sigma_out, [0, 2,3,1])           
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out        
        
class VDP_MaxPooling(keras.layers.Layer):
    """VDP_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='SAME'):
        super(VDP_MaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
    def call(self, mu_in, Sigma_in):
        batch_size = mu_in.shape[0] #shape=[batch_size, im_size, im_size, num_channel] 
        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1] 
        mu_out, argmax_out = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, self.pooling_size, self.pooling_size, 1], strides=[1, self.pooling_stride, self.pooling_stride, 1], padding=self.pooling_pad) #shape=[batch_zise, new_size,new_size,num_channel]
        hw_out = mu_out.shape[1]
        argmax1= tf.transpose(argmax_out, [0, 3, 1, 2])
        argmax2 = tf.reshape(argmax1,[batch_size, num_channel, -1])#shape=[batch_size, num_channel, new_size*new_size]        
        x_index = tf.math.floormod(tf.compat.v1.floor_div(argmax2,tf.constant(num_channel,shape=[batch_size,num_channel, hw_out*hw_out], dtype='int64')), tf.constant(hw_in ,shape=[batch_size,num_channel, hw_out*hw_out], dtype='int64'))     
        aux = tf.compat.v1.floor_div(tf.compat.v1.floor_div(argmax2,tf.constant(num_channel,shape=[batch_size,num_channel, hw_out*hw_out], dtype='int64')),tf.constant(hw_in,shape=[batch_size,num_channel, hw_out*hw_out], dtype='int64'))    
        y_index = tf.math.floormod(aux,tf.constant(hw_in,shape=[batch_size,num_channel,hw_out*hw_out], dtype='int64'))
        index = tf.multiply(y_index,hw_in) + x_index # shape=[batch_size, num_channel,new_size*new_size]        
        Sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2])#shape=[batch_size,num_channel,im_size*im_size, im_size*im_size] 
        gath1 = tf.gather(Sigma_in1, index, batch_dims = 2, axis=2)
        Sigma_out = tf.gather(gath1, index, batch_dims = 2, axis=-1)#shape=[batch_size,num_channel,new_size*new_size, new_size*new_size]
        Sigma_out = tf.transpose(Sigma_out, [0, 2,3, 1 ])#shape=[batch_size,new_size*new_size, new_size*new_size, num_channel]
        return mu_out, Sigma_out
        
        
class VDP_Flatten_and_FC(keras.layers.Layer):   
    def __init__(self, units):
        super(VDP_Flatten_and_FC, self).__init__()
        self.units = units                
    def build(self, input_shape):
        def sigma_regularizer_fc(x):      
            f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
            return (input_shape[1]*input_shape[2]*input_shape[-1])*tf.reduce_mean(-1. - tf.math.log(f_s) + f_s )
        ini_sigma = -2.2
        #min_sigma = -4.5
        tau = 1.# /(input_shape[1]*input_shape[2]*input_shape[-1] )      
        self.w_mu = self.add_weight(name = 'w_mu', shape=(input_shape[1]*input_shape[2]*input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None), regularizer=tf.keras.regularizers.l2(tau),#l1_l2(l1=tau, l2=tau), 
            trainable=True,
        )
        self.w_sigma = self.add_weight(name = 'w_sigma',
            shape=(self.units,),
            initializer= tf.constant_initializer(ini_sigma) , regularizer=sigma_regularizer_fc, 
            trainable=True, #tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None) 
        )    
    def call(self, mu_in, Sigma_in): 
        batch_size = mu_in.shape[0] #shape=[batch_size, im_size, im_size, num_channel] 
        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1]   
        mu_flatt = tf.reshape(mu_in, [batch_size, -1]) #shape=[batch_size, im_size*im_size*num_channel]           
        mu_out = tf.matmul(mu_flatt, self.w_mu) 
        fc_weight_mu1 = tf.reshape(self.w_mu, [num_channel, hw_in*hw_in ,self.units]) #shape=[num_channel, new_size*new_size, units]
        fc_weight_mu1T = tf.transpose(fc_weight_mu1,[0,2,1]) #shape=[num_channel,units,new_size*new_size]
        sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2]) #shape=[batch_size, num_channel, new_size*new_size, new_size*new_size]
        Sigma_1 = tf.matmul(tf.matmul(fc_weight_mu1T, sigma_in1), fc_weight_mu1 )#shape=[batch_size, num_channel, units, units]
        Sigma_1 = tf.math.reduce_sum(Sigma_1, axis=1) #shape=[batch_size, units, units]
        diag_elements = tf.linalg.trace(sigma_in1) #shape=[batch_size, num_channel]     
        tr_sigma_b =tf.math.reduce_sum(diag_elements,axis=1, keepdims=True) #shape=[batch_size, 1]

        tr_sigma_h_sigma_b = tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), tr_sigma_b ) # shape=[batch_size, units] 
        Sigma_2 = tf.linalg.diag(tr_sigma_h_sigma_b)# shape=[batch_size, units, units]
        mu_bT_mu_b = tf.math.reduce_sum(tf.multiply(mu_flatt, mu_flatt),axis=1, keepdims=True)  #shape=[batch_size, 1]
        mu_bT_sigma_h_mu_b = tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), mu_bT_mu_b) # shape=[batch_size, units] 
        Sigma_3 = tf.linalg.diag(mu_bT_sigma_h_mu_b)# shape=[batch_size, units, units]     
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3 
        
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)  
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out                
        		
class mysoftmax(keras.layers.Layer):
    """Mysoftmax"""
    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.softmax(mu_in)
        pp1 = tf.expand_dims(mu_out, axis=2)
        pp2 = tf.expand_dims(mu_out, axis=1)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        Sigma_out = tf.matmul(grad, tf.matmul(Sigma_in, tf.transpose(grad, perm=[0, 2, 1])))
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)        
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out	

class VDPReLU(keras.layers.Layer):
    def __init__(self):
        super(VDPReLU, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        with tf.GradientTape() as g:
          g.watch(mu_in)
          out = tf.nn.relu(mu_in)
        gradi = g.gradient(out, mu_in) 
        Sigma_out = activation_function_Sigma(gradi, Sigma_in)        
        return mu_out, Sigma_out    

def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):
    NS = tf.linalg.diag(tf.constant(1e-3, shape=[batch_size, num_labels]))#shape=[batch_size, num_labels, num_labels]
    I = tf.eye(num_labels, batch_shape=[batch_size])
    y_pred_sd_ns = y_pred_sd + NS
    y_pred_sd_inv = tf.linalg.solve(y_pred_sd_ns, I)
    mu_ = y_pred_mean - y_test #shape=[batch_size, num_labels]
    mu_sigma = tf.matmul(tf.expand_dims(mu_, axis=1) ,  y_pred_sd_inv)  #shape=[batch_size, 1, num_labels]
    ms1 = tf.math.reduce_mean(tf.squeeze(tf.matmul(mu_sigma , tf.expand_dims(mu_, axis=2))) )
    ms2 = tf.math.reduce_mean(tf.squeeze(tf.linalg.slogdet(y_pred_sd_ns)[1]))
    ms = tf.math.reduce_mean(ms1 + ms2)
    return ms    
            
class Density_prop_CNN(tf.keras.Model):
  def __init__(self, kernel_size,num_kernel, pooling_size, pooling_stride, pooling_pad, units, name=None):
    super(Density_prop_CNN, self).__init__()
    self.kernel_size = kernel_size
    self.num_kernel = num_kernel
    self.pooling_size = pooling_size
    self.pooling_stride = pooling_stride
    self.pooling_pad = pooling_pad
    self.units = units
    self.conv_1 = VDP_first_Conv(kernel_size=self.kernel_size[0], kernel_num=self.num_kernel[0], kernel_stride=1, padding="VALID")
    self.relu_1 = VDPReLU()
    self.maxpooling_1 = VDP_MaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0], pooling_pad=self.pooling_pad)
    self.fc_1 = VDP_Flatten_and_FC(self.units)   
    self.mysoftma = mysoftmax()
    
  def call(self, inputs, training=True):
    mu1, sigma1 = self.conv_1(inputs) 
    mu2, sigma2 = self.relu_1(mu1, sigma1) 
    mu3, sigma3 = self.maxpooling_1(mu2, sigma2) 
    mu4, sigma4 = self.fc_1(mu3, sigma3)   
    mu_out, Sigma_out = self.mysoftma(mu4, sigma4)    
    Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
    Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)        
    return mu_out, Sigma_out    
      
    
def main_function(input_dim=28, num_kernels=[32], kernels_size=[5], maxpooling_size=[2], maxpooling_stride=[2], maxpooling_pad='SAME', class_num=10 , batch_size=100,
        epochs =20, lr=0.001, lr_end = 0.0001, kl_factor = 0.01,
        Random_noise=True, gaussain_noise_std=0.5, Adversarial_noise=False, epsilon = 0, adversary_target_cls=3, Targeted=False,
        Training = False, continue_training = False,  saved_model_epochs=50):   
       
    PATH = './saved_models/VDP_cnn_epoch_{}/'.format( epochs)   
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    one_hot_y_train = tf.one_hot(y_train.astype(np.float32), depth=class_num)
    one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=class_num)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)
    
    cnn_model = Density_prop_CNN(kernel_size=kernels_size,num_kernel=num_kernels, pooling_size=maxpooling_size, pooling_stride=maxpooling_stride, pooling_pad=maxpooling_pad, units=class_num, name = 'vdp_cnn')       
    num_train_steps = epochs * int(x_train.shape[0] /batch_size)
#    step = min(step, decay_steps)
#    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,  decay_steps=num_train_steps,  end_learning_rate=lr_end, power=2.)     
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)#, clipnorm=1.0)
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            mu_out, sigma = cnn_model(x, training=True)  
            cnn_model.trainable = True         
            loss_final = nll_gaussian(y, mu_out,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+5),
                                   clip_value_max=tf.constant(1e+5)), class_num , batch_size)
            regularization_loss=tf.math.add_n(cnn_model.losses)             
            loss = 0.5 * (loss_final + kl_factor*regularization_loss )           
            gradients = tape.gradient(loss, cnn_model.trainable_weights)        
            gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
            gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, cnn_model.trainable_weights))       
        return loss, mu_out, sigma, gradients
    @tf.function
    def validation_on_batch(x, y):                     
        mu_out, sigma = cnn_model(x, training=False) 
        cnn_model.trainable = False              
        vloss = nll_gaussian(y, mu_out,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+4),
                                           clip_value_max=tf.constant(1e+4)), class_num , batch_size)                                           
        regularization_loss=tf.math.add_n(cnn_model.losses)
        total_vloss = 0.5 *(vloss + kl_factor*regularization_loss)    
        return total_vloss, mu_out, sigma
    @tf.function
    def test_on_batch(x, y):  
        cnn_model.trainable = False                    
        mu_out, sigma = cnn_model(x, training=False)            
        return mu_out, sigma
    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            cnn_model.trainable = False 
            prediction, sigma = cnn_model(input_image) 
            loss_final = nll_gaussian(input_label, prediction,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+4),
                                   clip_value_max=tf.constant(1e+3)), class_num , batch_size)                         
            loss = 0.5 * loss_final 
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad         
    if Training:
      #  wandb.init(entity = "dimah", project="VDP_CNN_mnist_epochs_{}_lr_{}_lambda02_test1".format(epochs, lr))
        if continue_training:
            saved_model_path = './saved_models/VDP_cnn_epoch_{}/'.format(saved_model_epochs)
            cnn_model.load_weights(saved_model_path + 'vdp_cnn_model')
        train_acc = np.zeros(epochs) 
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        
        start = timeit.default_timer()       
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)           
            acc1 = 0
            acc_valid1 = 0 
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0           
            #-------------Training--------------------
            for step, (x, y) in enumerate(tr_dataset):                         
                update_progress(step/int(x_train.shape[0]/(batch_size)) )                
                loss, mu_out, sigma, gradients = train_on_batch(x, y)                      
                err1+= loss.numpy() 
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))                            
                acc1+=accuracy.numpy()                 
                if step % 50 == 0:
                    print('\n gradient', np.mean(gradients[0].numpy()))
                    print('\n Matrix Norm', np.mean(sigma))
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy so far: %.3f" % float(acc1/(tr_no_steps + 1.)))                                                                   
                tr_no_steps+=1 
##                wandb.log({"Average Variance value": tf.reduce_mean(sigma).numpy(),                      
##                             "Total Training Loss": loss.numpy() ,
##                             "Training Accuracy per minibatch": accuracy.numpy() ,                                                 
##                             "gradient per minibatch": np.mean(gradients[0]),                              
##                             'epoch': epoch
##                    })        
            train_acc[epoch] = acc1/tr_no_steps
            train_err[epoch] = err1/tr_no_steps        
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])         
            #---------------Validation----------------------                  
            for step, (x, y) in enumerate(val_dataset):               
                update_progress(step / int(x_test.shape[0] / (batch_size)) )   
                total_vloss, mu_out, sigma   = validation_on_batch(x, y)                
                err_valid1+= total_vloss.numpy()                               
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1),tf.math.argmax(y,axis=-1))
                va_accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_valid1+=va_accuracy.numpy() 
                
                if step % 50 == 0:                   
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy so far: %.3f" % va_accuracy)               
                va_no_steps+=1
##                wandb.log({"Average Variance value (validation Set)": tf.reduce_mean(sigma).numpy(),                         
##                               "Total Validation Loss": total_vloss.numpy() ,                              
##                               "Validation Acuracy per minibatch": va_accuracy.numpy()                                                         
##                                })          
            valid_acc[epoch] = acc_valid1/va_no_steps      
            valid_error[epoch] = err_valid1/va_no_steps
            stop = timeit.default_timer()
            cnn_model.save_weights(PATH + 'vdp_cnn_model')   
##            wandb.log({"Average Training Loss":  train_err[epoch],                        
##                        "Average Training Accuracy": train_acc[epoch],                                            
##                        "Average Validation Loss": valid_error[epoch],                                           
##                        "Average Validation Accuracy": valid_acc[epoch],                                           
##                        'epoch': epoch
##                       }) 
            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch])
            print('Validation Acc  ', valid_acc[epoch])           
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])           
        #-----------------End Training--------------------------             
        cnn_model.save_weights(PATH + 'vdp_cnn_model')        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Density Propagation CNN on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_CNN_on_MNIST_Data_acc.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')            
            plt.title("Density Propagation CNN on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_CNN_on_MNIST_Data_error.png')
            plt.close(fig)
        
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')         
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)                                                   
        f.close()                  
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))         
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
                textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))
                    
                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc[epoch])))
                
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    elif(Random_noise):
        test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        cnn_model.load_weights(PATH + 'vdp_cnn_model') 
        test_no_steps = 0        
        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
            true_x[test_no_steps, :, :, :,:] = x
            true_y[test_no_steps, :, :] = y
            if Random_noise:
                noise = tf.random.normal(shape = [batch_size, input_dim, input_dim, 1], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                x = x +  noise
            mu_out, sigma   = test_on_batch(x, y)              
            mu_out_[test_no_steps,:,:] = mu_out
            sigma_[test_no_steps, :, :, :]= sigma             
            corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
            accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()
            if step % 100 == 0:
                print("Total running accuracy so far: %.3f" % accuracy.numpy())             
            test_no_steps+=1      
             
        test_acc = np.mean(acc_test)          
        print('Test accuracy : ', test_acc)                  
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')      
        pickle.dump([mu_out_, sigma_, test_acc ], pf)                                                  
        pf.close()
        
        var = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        for i in range(int(x_test.shape[0] / (batch_size))):
            for j in range(batch_size):
                noise = tf.random.normal(shape = [input_dim, input_dim, 1], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:,:, :]))/np.sum( np.square(noise) ))
         
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))   
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : "+ str( test_acc)) 
        textfile.write("\n Output Variance: "+ str(np.mean(var)))   
                         
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))   
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))           
        textfile.write("\n---------------------------------")    
        textfile.close()
    elif(Adversarial_noise):
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(epsilon)            
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)              
        cnn_model.load_weights(PATH + 'vdp_cnn_model')       
        test_no_steps = 0        
        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
            true_x[test_no_steps, :, :, :,:] = x
            true_y[test_no_steps, :, :] = y
            
            if Targeted:
                y_true_batch = np.zeros_like(y)
                y_true_batch[:, adversary_target_cls] = 1.0            
                adv_perturbations[test_no_steps, :, :, :,:] = create_adversarial_pattern(x, y_true_batch)
            else:
                adv_perturbations[test_no_steps, :, :, :,:] = create_adversarial_pattern(x, y)
            adv_x = x + epsilon*adv_perturbations[test_no_steps, :, :, :,:] 
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0) 
            
            mu_out, sigma   = test_on_batch(adv_x, y)           
            mu_out_[test_no_steps,:,:] = mu_out
            sigma_[test_no_steps, :, :, :]= sigma             
            corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
            accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
            acc_test[test_no_steps]=accuracy.numpy()
            if step % 10 == 0:
                print("Total running accuracy so far: %.3f" % accuracy.numpy())             
            test_no_steps+=1               
            
        test_acc = np.mean(acc_test)         
        print('Test accuracy : ', test_acc)                       
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')            
        pickle.dump([mu_out_, sigma_,  adv_perturbations, test_acc], pf)                                                
        pf.close()
        
        var = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        for i in range(int(x_test.shape[0] /batch_size)):
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :,:]))/np.sum( np.square(epsilon*adv_perturbations[i, j, :, :, :]  ) ))
         
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))         
        
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))      
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc))  
        textfile.write("\n Output Variance: "+ str(np.mean(var)))                     
        textfile.write("\n---------------------------------")
        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))                   
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))    
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
        textfile.write("\n---------------------------------")    
        textfile.close()   
if __name__ == '__main__':
    main_function() 
