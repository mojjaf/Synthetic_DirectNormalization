import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D,Conv2D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D,SpatialDropout3D,Activation,multiply,Add,BatchNormalization
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv3D, Input,MaxPooling2D, MaxPooling3D, Dropout, concatenate, UpSampling3D,UpSampling2D

from mainargs import get_args

args = get_args()

IMG_WIDTH = args.image_size
IMG_HEIGHT = args.image_size
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS=args.output_channel
CHANNELS=args.input_channel

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())

    return result



def Generator2D():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
    filter=64

    down_stack = [
    downsample(filter, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(filter*2, 4),  # (batch_size, 64, 64, 128)
    downsample(filter*4, 4),  # (batch_size, 32, 32, 256)
    downsample(filter*8, 4),  # (batch_size, 16, 16, 512)
    downsample(filter*8, 4),  # (batch_size, 8, 8, 512)
    downsample(filter*8, 4),  # (batch_size, 4, 4, 512)
    downsample(filter*8, 4),  # (batch_size, 2, 2, 512)
    downsample(filter*8, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(filter*8, 4),  # (batch_size, 16, 16, 1024)
    upsample(filter*4, 4),  # (batch_size, 32, 32, 512)
    upsample(filter*2, 4),  # (batch_size, 64, 64, 256)
    upsample(filter, 4),  # (batch_size, 128, 128, 128)
    ]
    
  
    

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

     # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
def Discriminator2D():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
    kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def attention_gate(g, x, output_channel, padding='same'):
    g1 = tf.keras.layers.Conv2D(output_channel, kernel_size=1, strides=1, padding=padding)(g)
    g1 = tf.keras.layers.BatchNormalization()(g1)
    x1 = tf.keras.layers.Conv2D(output_channel, kernel_size=1, strides=1, padding=padding)(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    psi = tf.keras.layers.Activation("relu")(Add()([g1, x1]))
    psi = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding=padding)(psi)
    psi = tf.keras.layers.BatchNormalization()(psi)
    psi = tf.keras.layers.Activation("sigmoid")(psi)
    return tf.keras.layers.multiply([x, psi])


def Generator_AG2D():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
    filter=64

    down_stack = [
    downsample(filter, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(filter*2, 4),  # (batch_size, 64, 64, 128)
    downsample(filter*4, 4),  # (batch_size, 32, 32, 256)
    downsample(filter*8, 4),  # (batch_size, 16, 16, 512)
    downsample(filter*8, 4),  # (batch_size, 8, 8, 512)
    downsample(filter*8, 4),  # (batch_size, 4, 4, 512)
    downsample(filter*8, 4),  # (batch_size, 2, 2, 512)
    downsample(filter*8, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(filter*8, 4),  # (batch_size, 16, 16, 1024)
    upsample(filter*4, 4),  # (batch_size, 32, 32, 512)
    upsample(filter*2, 4),  # (batch_size, 64, 64, 256)
    upsample(filter, 4),  # (batch_size, 128, 128, 128)
    ]
    
  
    
    filters=[512,512,512,512,256,128,64]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

     # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip, filt in zip(up_stack, skips, filters):
        x = up(x)
        x = attention_gate(skip,x, filt)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)





   
def Generator3D():
    '''
    Generator model
    '''
    def encoder_step(layer, Nf, ks, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
            #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        return x

    def bottlenek(layer, Nf, ks):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(4):
            y = Conv3D(Nf, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
            x = InstanceNormalization()(y)
            #x= tf.keras.layers.BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Concatenate()([x, y])

        return x

    def decoder_step(layer, layer_to_concatenate, Nf, ks):
        x=UpSampling3D(size = (2,2,2))(layer)
        #x = Conv3DTranspose(Nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(layer)
        x = InstanceNormalization()(x)
        #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Concatenate()([x, layer_to_concatenate])
        x = Dropout(0.2)(x)
        return x

    layers_to_concatenate = []
    inputs = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='input_image')
    Nfilter_start = 64
    depth = 4
    ks = 3
    x = inputs

    # encoder
    for d in range(depth-1):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks, False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks)
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottlenek(x, Nfilter_start*np.power(2,depth-1), ks)

    # decoder
    for d in range(depth-2, -1, -1): 
        x = decoder_step(x, layers_to_concatenate.pop(), Nfilter_start*np.power(2,d), ks)

    # classifier
    last = Conv3DTranspose(1, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal', activation='tanh', name='output_generator')(x)
    #print(tf.shape(last))
    #last=
    return Model(inputs=inputs, outputs=last, name='Generator')

def Discriminator3D():
    '''
    Discriminator model
    '''

    inputs = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='input_image')
    targets = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='target_image')
    Nfilter_start = 32
    depth = 4
    ks = 3
    

    def encoder_step(layer, Nf, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
            #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        return x

    x = Concatenate()([inputs, targets])

    for d in range(depth):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d))
            
    x = ZeroPadding3D()(x)
    x = Conv3D(Nfilter_start*(2**depth), ks, strides=1, padding='valid', kernel_initializer='he_normal')(x) 
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
      
    x = ZeroPadding3D()(x)
    last = Conv3D(1, ks, strides=1, padding='valid', kernel_initializer='he_normal', name='output_discriminator')(x) 

    return Model(inputs=[inputs, targets], outputs=last, name='Discriminator')


def attention_3D(g, x, output_channel, padding='same'):
    g1 = Conv3D(output_channel, kernel_size=1, strides=1, padding=padding)(g)
    g1 = BatchNormalization()(g1)
    x1 = Conv3D(output_channel, kernel_size=1, strides=1, padding=padding)(x)
    x1 = BatchNormalization()(x1)
    psi = Activation("relu")(Add()([g1, x1]))
    psi = Conv3D(1, kernel_size=1, strides=1, padding=padding)(psi)
    psi = BatchNormalization()(psi)
    psi = Activation("sigmoid")(psi)
    return multiply([x, psi])


def Generator_Attention3D():
    '''
    Generator model
    '''
    def encoder_step(layer, Nf, ks, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
            #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = SpatialDropout3D(0.2)(x)

        return x

    def bottlenek(layer, Nf, ks):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(4):
            y = Conv3D(Nf, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
            x = InstanceNormalization()(y)
            #x= tf.keras.layers.BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Concatenate()([x, y])

        return x

    def decoder_step(layer, layer_to_concatenate, Nf, ks):
        x = Conv3DTranspose(Nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(layer)
        x = attention_3D(layer_to_concatenate,x,Nf)
        x = InstanceNormalization()(x)
        #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Concatenate()([x, layer_to_concatenate])
        x = SpatialDropout3D(0.2)(x)
        return x

    layers_to_concatenate = []
    inputs = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='input_image')
    Nfilter_start = 32
    depth = 4
    ks = 3
    x = inputs

    # encoder
    for d in range(depth-1):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks, False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks)
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottlenek(x, Nfilter_start*np.power(2,depth-1), ks)

    # decoder
    for d in range(depth-2, -1, -1): 
        x = decoder_step(x, layers_to_concatenate.pop(), Nfilter_start*np.power(2,d), ks)

    # classifier
    last = Conv3DTranspose(1, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal', activation='tanh', name='output_generator')(x)
    #print(tf.shape(last))
    #last=
    return Model(inputs=inputs, outputs=last, name='Generator_AttentionUnet')




def Unet3D_Generator():
    inputs=Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='input_image')
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same',data_format="channels_last")(inputs)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(512, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv3D(512, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(256, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=-1)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv3D(128, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv3D(64, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv3D(32, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv3D(1, 1, activation = 'tanh')(conv9)
    model = Model(inputs=inputs, outputs = conv10)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def Unet2D_Generator():
    inputs=Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS), name='input_image')
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same',data_format="channels_last")(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=-1)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = 'tanh')(conv9)
    model = Model(inputs=inputs, outputs = conv10)
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model