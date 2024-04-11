#! /usr/local/bin/path_to_python

####################################
### MLOps: BUILDING A SIMPLE APP ###
### MAKE INFERENCE               ###
### by: OAMEED NOAKOASTEEN       ###
####################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy      as np
import tensorflow as tf

def initialize_run():
    dir_parent      = "run"
    dir_current_run = os.listdir(dir_parent)[0]
    paths           = [os.path.join(dir_parent,dir_current_run,"checkpoints"),
                       os.path.join(dir_parent,dir_current_run,"predictions") ]
    return paths

def get_data():
    _, (x,_) = tf.keras.datasets.mnist.load_data()
    x        = (x-np.amin(x))/(np.amax(x)-np.amin(x))
    if len(x.shape) == 3:
     x       = np.expand_dims(x,axis=3)
    return x.astype('float32')

def get_a_sample_of_data():
    x   = get_data()
    rng = np.random.default_rng(2024)
    idx = sorted(list(rng.integers(0,x.shape[0],100)))
    x   = x[idx]
    return x

def plotter(x,filename):
    from matplotlib import pyplot as plt
    size  = 10
    cmap  = 'viridis'
    if x.shape[-1] == 1:
     x    = np.squeeze(x)
     cmap = 'gray'
    fig  = plt.figure()
    for  i in range(size):
     for j in range(size):
      ax = plt.subplot(size,size,i*size+(j+1))
      ax.axis('off')
      ax.imshow(x[i*size+j], cmap=cmap)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0.075)
    plt.savefig(filename,format='png')

def main():
    paths = initialize_run()
    
    x     = get_a_sample_of_data()
    
    model = tf.keras.models.load_model(paths[0])
    
    x     = model(x, training=False).numpy() 
    
    plotter(x, os.path.join(paths[1],"predictions"+".png"))
    
    print("Inference Finished!")

if __name__ == "__main__":
 main()

