#! /usr/local/bin/path_to_python

####################################
### MLOps: BUILDING A SIMPLE APP ###
### TRAIN AN AUTO-ENCODER        ###
### by: OAMEED NOAKOASTEEN       ###
####################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy      as np
import tensorflow as tf

def initialize_run():
    # paths [0]   : checkpoints
    # paths [1]   : logs
    # paths [2]   : predictions
    # paths [3]   : tfrecords
    # params[0][0]: number of epochs
    # params[0][1]: batch size
    # params[1][*]: data shape
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, default=2 )
    parser.add_argument('-b', type=int, default=32)
    args   = parser.parse_args()
    params = [[args.e, args.b]]
    paths  = get_paths()
    return params, paths

def get_paths():
    def get_current_run_name():
        import datetime
        now = datetime.datetime.now()
        now = [now.year, now.month, now.day, now.hour, now.minute, now.second]
        return "_".join([str(x) for x in now])
    import shutil
    dir_parent      =  "run"
    dir_current_run = get_current_run_name()
    dir_subdirs     = ["checkpoints", "logs", "predictions","tfrecords"]
    paths           = [os.path.join(dir_parent,dir_current_run,x) for x in dir_subdirs]
    shutil.rmtree(dir_parent, ignore_errors=True)
    for path in paths:
     os.makedirs(path)
    return paths

def get_data(mode):
    if mode == 'train':
     (x,y), _     = tf.keras.datasets.mnist.load_data()
    else:
     if mode == 'validation':
      _   , (x,y) = tf.keras.datasets.mnist.load_data()
    x = (x-np.amin(x))/(np.amax(x)-np.amin(x))
    if len(x.shape)==3:
     x = np.expand_dims(x,axis=3)
    return x.astype('float32'),y

def save_data_to_tfrecords_format(params,paths):
    def get_shape(x):
        return [i for i in x.shape[1:]]
    def serialize_example(x,y):
        feature={'X':   tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tobytes()])), 
                 'Y':   tf.train.Feature(int64_list=tf.train.Int64List(value=[y          ])) }
        example_proto = tf.train.Example(features  =tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    def write_serialized_example(x,y,filename):
        with tf.io.TFRecordWriter(filename) as writer:
         for i in range(x.shape[0]):
          example = serialize_example(x[i],y[i])
          writer.write(example)
    x,y      = get_data('train')
    params.append(get_shape(x))
    filename = os.path.join(paths[3],'data'+'.tfrecords')
    write_serialized_example(x,y,filename)

def read_tfrecords(params,paths):
    def get_filenames(path):
        filenames = []
        for file in os.listdir(path):
         if not file.startswith('.'):
          filenames.append(os.path.join(path,file)) 
        return filenames
    feature = {'X': tf.io.FixedLenFeature([],tf.string), 
               'Y': tf.io.FixedLenFeature([],tf.int64 ) }
    def parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto,feature )
        x              = tf.io.decode_raw(parsed_example['X'],tf.float32)
        x.set_shape([np.prod(params[1])])
        x              = tf.reshape(x,params[1])
        y              = parsed_example['Y']
        return x,y
    filenames = get_filenames(paths[3])
    dataset   = tf.data.TFRecordDataset(filenames                                  )
    dataset   = dataset.map            (parse_function                             )
    dataset   = dataset.batch          (params[0][1], drop_remainder          =True)
    dataset   = dataset.shuffle        (10000       , reshuffle_each_iteration=True)
    return dataset

def get_encoder(params):
    def layer_conv2d(x,size_out,size_filter,strides,padding='same'):
        initializer = tf.keras.initializers.he_normal()
        x           = tf.keras.layers.Conv2D         (size_out                         ,  
                                                      size_filter                      ,
                                                      strides            = strides     ,
                                                      padding            = padding     , 
                                                      activation         = None        ,
                                                      kernel_initializer = initializer  )(x)
        return x
    X = tf.keras.Input(shape=params[1]  )
    x = layer_conv2d  (X,64 ,[5,5],[2,2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU              ()(x)
    x = layer_conv2d  (x,128,[3,3],[2,2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU              ()(x)
    return tf.keras.Model(inputs=X,outputs=x)

def get_decoder(params,input_shape):
    def layer_conv2dtrans(x,size_out,size_filter,strides,padding='same'):
        initializer = tf.keras.initializers.he_normal()
        x           = tf.keras.layers.Conv2DTranspose(size_out                         ,
                                                      size_filter                      ,
                                                      strides            = strides     ,
                                                      padding            = padding     ,
                                                      activation         = None        ,
                                                      kernel_initializer = initializer  )(x)
        return x
    X = tf.keras.Input   (shape=input_shape)
    x = layer_conv2dtrans(X,64          ,[3,3],[2,2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU              ()(x)
    x = layer_conv2dtrans(x,params[1][2],[3,3],[2,2])
    return tf.keras.Model(inputs=X,outputs=x)

class autoencoder(tf.keras.Model):
    def __init__(self,encoder,decoder,params,paths):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.params  = params
        self.paths   = paths

    def compile(self,optimizer,loss):
        super().compile()
        self.optimizer = optimizer
        self.loss      = loss

    def call(self,x,training=None):
        x = self.encoder(x,training=training)
        x = self.decoder(x,training=training)
        return x

    def train_step(self,data):
        x,y = data
        with tf.GradientTape() as tape:
         predictions = self(x,training=True)
         loss        = self.loss(x,predictions)
        gradients    = tape.gradient  (    loss     , self.trainable_weights )
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {'loss':loss}

class callback_custom_ckpt   (tf.keras.callbacks.Callback):
    def on_train_end(self, epoch, logs=None):
        tf.keras.models.save_model(self.model ,self.model.paths[0])

class callback_custom_monitor(tf.keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def plotter(self,x,cmap):
        from matplotlib import pyplot as plt
        size = 10
        fig  = plt.figure()
        for  i in range(size):
         for j in range(size):
          ax = plt.subplot(size,size,i*size+(j+1))
          ax.axis('off')
          ax.imshow(x[i*size+j], cmap=cmap)
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0.075)
        return fig        

    def plot_to_image_converter(self,fig):
        import                           io
        from matplotlib import pyplot as plt
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close  (fig) 
        buf.seek   (0)
        img = tf.image.decode_png(buf.getvalue(), channels=0)
        img = tf.expand_dims     (img           ,          0)
        return img

    def on_epoch_begin(self, epoch, logs=None):
        x,_    = self.data
        rng    = np.random.default_rng(2024)
        idx    = sorted(list(rng.integers(0,x.shape[0],100)))
        x      = x[idx]
        x      = self.model(x, training=False).numpy()
        cmap   = 'viridis'
        if x.shape[-1]==1:
         x     = np.squeeze(x)
         cmap  = 'gray'
        fig    = self.plotter(x,cmap)
        img    = self.plot_to_image_converter(fig)
        writer = tf.summary.create_file_writer(self.model.paths[1])
        with writer.as_default():
         tf.summary.image("Reconstructed Images", img, step=epoch)

class callback_custom_history(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.metric_one = []

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.metric_one.append(logs[keys[0]])

    def plotter(self,x,ylabel,filename):
        from matplotlib import pyplot as plt
        x_axis = np.array([i for i in range(len(x))])/1e3
        fig,ax = plt.subplots()
        ax.plot      (x_axis,x, 'b', linewidth = 2, label=ylabel)
        ax.set_xlabel(' Iterations (K) ')
        ax.legend    ()
        ax.grid      ()
        plt.savefig(filename,format='png')

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        self.plotter(self.metric_one                                 ,
                     keys[0]                                         ,
                     os.path.join(self.model.paths[1],keys[0]+'.png') )

def main():
    params, paths      = initialize_run()
    
    save_data_to_tfrecords_format      (params,paths)
    data_train         = read_tfrecords(params,paths)
    data_validation    = get_data('validation')
    
    encoder            = get_encoder(params)
    encoder_out_shape  = [x for x in encoder.layers[-1].output_shape[1:]]
    decoder            = get_decoder(params,encoder_out_shape)
    
    model              = autoencoder(encoder = encoder,
                                     decoder = decoder,
                                     params  = params ,
                                     paths   = paths   )
    model.compile(optimizer = tf.keras.optimizers.Adam        (),
                  loss      = tf.keras.losses.MeanSquaredError() )
    
    callbacks          = [callback_custom_ckpt()]
    if False:
     callbacks.append(callback_custom_monitor       (data = data_validation    ),
                      callback_custom_history       ()                          ,
                      tf.keras.callbacks.TensorBoard(log_dir        = paths[1], 
                                                     histogram_freq = 1       , 
                                                     update_freq    ='batch'   ) )
    
    model.fit    (x               = data_train     ,
                  epochs          = params[0][0]   ,
                  validation_data = data_validation,
                  callbacks       = callbacks      ,
                  verbose         = 2               )
    
    print("Training Finished!")

if __name__ == "__main__":
 main()

