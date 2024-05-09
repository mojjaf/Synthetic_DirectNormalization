#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf

print("\n * START CONFIGURATION* \n")

############################## Define the number of Physical and Virtual GPUs #############

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[2:3], 'GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20000),
             tf.config.LogicalDeviceConfiguration(memory_limit=20000)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
        print(e)


print("\n * GPU Setup compelted... * \n")


################################# load libraries #######################################
import os 
import numpy as np 
import datetime
import time
import sys
from IPython import display
from utils import resize, generate_images,normalise_per_channel,normalize_image
from evaluation import evaluate_model
from losses import generator_loss, discriminator_loss
from mainargs import get_args
from matplotlib import pyplot as plt
sys.path.insert(0, './pymiil/')
from models import Generator2D, Discriminator2D
from pymiil.miil.io import load_cuda_vox,write_cuda_vox
from utils import progressbar, resize, normalize_tensor
from pathlib import Path
import cv2
# In[3]:


############################## Basic configurations ##################################
dataset= "DirectNorm_"
mode='slice_by_slice'  #2.5 D or single (2D)
data_version='LC_sensitivityimages'
experiment = "/experiments/Pix2Pix_Normalization_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"  # 
model_name = "model_25D_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/" 
print(f"\nExperiment: {experiment}\n")
args = get_args()
project_dir = args.main_dir
#data_dir=args.data_dir

experiment_dir = project_dir+experiment+mode+data_version

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir) 
    
models_dir = experiment_dir+model_name
if not os.path.exists(models_dir):
    os.makedirs(models_dir) 
    
output_preds_2D = experiment_dir+'pix2pix2D_predictions/'

if not os.path.exists(output_preds_2D):
    os.makedirs(output_preds_2D)

PATH=args.data_dir
IMG_WIDTH = args.image_size
IMG_HEIGHT = args.image_size
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS = args.output_channel
print(INPUT_CHANNELS, OUTPUT_CHANNELS)

print(IMG_HEIGHT, IMG_WIDTH)


def pairmatch_vox_files_from_directory_2D(image_dirA,image_dirB, height, width):
    input_image_list= []#np.zeros((ids,image_size, image_size, 3), dtype=float)
    real_image_list= []#np.zeros((ids,image_size, image_size, 3), dtype=float)
    image_suffixes = (".h5",".mat",".jpeg", ".jpg", ".png", ".tif",".vox")
    image_filenames = [p for p in Path(image_dirA).glob("**/*") if p.suffix.lower() in image_suffixes]
    progressbar(0, len(image_filenames))
    i=0
    for image_file in (os.listdir(image_dirA)):
        #print(image_file)
        ImgA= load_cuda_vox(os.path.join(image_dirA, image_file)) #input image
        ImgB= load_cuda_vox(image_dirB) #target image file name is equal to the input image file name
        zlen=ImgA.shape[2]
        for plane in range(zlen):
            ImgA_2D = ImgA[:, :, plane]
            ImgA_2D=np.expand_dims(ImgA_2D, axis=-1)
            ImgB_2D=ImgB[:,:,plane]
            ImgB_2D=np.expand_dims(ImgB_2D, axis=-1)
            input_image,real_image=resize(ImgA_2D, ImgB_2D, height, width)
            input_image,real_image=normalize_tensor(input_image,real_image)
            input_image_list.append(input_image)
            real_image_list.append(real_image)
            
        progressbar(i+1, len(image_filenames))
        i+=1
    return input_image_list,real_image_list


########################### Read Training dataset  ########################### 


image_res_='/home/mojjaf/Direct Normalization/Data/Training Data/Resolution'
image_res_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/Resolution_FullCount.vox'
input_res_,real_res_=pairmatch_vox_files_from_directory_2D(image_res_,image_res_target, IMG_WIDTH, IMG_HEIGHT)

image_con_='/home/mojjaf/Direct Normalization/Data/Training Data/Contrast'
image_con_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/Contrast_FullCount.vox'

input_con_,real_con_=pairmatch_vox_files_from_directory_2D(image_con_,image_con_target, IMG_WIDTH, IMG_HEIGHT)

image_cyl_='/home/mojjaf/Direct Normalization/Data/Training Data/Cylinder'
image_cyl_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/Clinder_FullCount.vox'

input_cyl_,real_cyl_=pairmatch_vox_files_from_directory_2D(image_cyl_,image_cyl_target, IMG_WIDTH, IMG_HEIGHT)


image_MiniRes_='/home/mojjaf/Direct Normalization/Data/Training Data/MiniResolution'
image_MiniRes_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/MiniResolution_FullCount.vox'

input_MiniRes_,real_MiniRes_=pairmatch_vox_files_from_directory_2D(image_MiniRes_,image_MiniRes_target, IMG_WIDTH, IMG_HEIGHT)

image_brain_='/home/mojjaf/Direct Normalization/Data/Training Data/BrainPET'
image_brain_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/BrainPET.vox'

input_brain_,real_brain_=pairmatch_vox_files_from_directory_2D(image_brain_,image_brain_target, IMG_WIDTH, IMG_HEIGHT)

image_ConV29_='/home/mojjaf/Direct Normalization/Data/Training Data/Contrast_v29'
image_ConV29_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/ContrastNew_fullcount.vox'

input_ConV29_,real_ConV29_=pairmatch_vox_files_from_directory_2D(image_ConV29_,image_ConV29_target, IMG_WIDTH, IMG_HEIGHT)

image_hoffman_='/home/mojjaf/Direct Normalization/Data/Training Data/HoffmanV3'
image_hoffman_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/BrainHoffmanV3_fullND.vox'

input_hoffman_,real_hoffman_=pairmatch_vox_files_from_directory_2D(image_hoffman_,image_hoffman_target, IMG_WIDTH, IMG_HEIGHT)



input_image_train=np.vstack([input_res_,input_con_,input_MiniRes_,input_cyl_,input_brain_,input_ConV29_,input_hoffman_])
real_image_train=np.vstack([real_res_,real_con_,real_MiniRes_,real_cyl_,real_brain_,real_ConV29_,real_hoffman_])

print(np.shape(input_image_train))


############################ Read Validation Dataset  ########################### 


image_val='/home/mojjaf/Direct Normalization/Data/Validation Data/LC Resolution'
image_val_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/Resolution_FullCount.vox'
input_image_val,real_image_val=pairmatch_vox_files_from_directory_2D(image_val,image_val_target, IMG_HEIGHT, IMG_WIDTH)


###########################  Create Training and Validation Tensors########################### 
BUFFER_SIZE = len(input_image_train)
BATCH_SIZE = args.batch_size

AUTOTUNE = tf.data.AUTOTUNE
with tf.device('/device:cpu:0'):
    datasetx = tf.data.Dataset.from_tensor_slices((input_image_train))

    datasety = tf.data.Dataset.from_tensor_slices((real_image_train))
    train_dataset = tf.data.Dataset.zip((datasetx, datasety))

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    
    


# In[10]:


with tf.device('/device:cpu:0'):
    datasetxVal = tf.data.Dataset.from_tensor_slices((input_image_val))
    datasetyVal = tf.data.Dataset.from_tensor_slices((real_image_val))
    val_dataset = tf.data.Dataset.zip((datasetxVal, datasetyVal)).batch(BATCH_SIZE)

print("\n * Validation dataset successfully created. * \n")


# In[11]:


def generate_images(model, input_image, target):
    prediction = model(input_image, training=True)
    plt.figure(figsize=(15, 15))
    print(np.shape(prediction))
    
    residual=np.abs(target[0]-prediction[0])

    display_list = [input_image[0,...,0], target[0], prediction[0],residual]
    title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Residual Image']
# set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        if i<3:
            plt.imshow(display_list[i],vmin=-1, vmax=1, cmap='jet')
            plt.colorbar(fraction=0.046, pad=0.04)
        else: 
            plt.imshow(display_list[i],vmin=0, vmax=1, cmap='RdBu_r')
            plt.colorbar(fraction=0.046, pad=0.04)
        
        
    
        plt.axis('off')
    plt.show()


# In[12]:



# In[9]:
########################### GAN MODELS configuration ########
generator = Generator2D()
discriminator = Discriminator2D()

learning_rate=args.lr

generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)#original 0.5
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

############################# TRAINING Configuration ########

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
       
print("\n * GAN models successfully configured... * \n")

import datetime
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def fit(train_ds, steps):
    example_input, example_target = next(iter(val_dataset.take(1)))
    start = time.time()
    epoch=0

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)
            #print("\n * TRAINING... * \n")

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target)
            print("\n * TRAINING... * \n")
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)
        
        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 20000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        # Save the model every 20k steps
        if (step + 1) % 10000 == 0:
            epoch+=1
            generator.save_weights('./gen_'+ str(epoch) + '.h5')
            print("\n * MODEL SAVED * \n")

    

#################################### Start Training 
print("\n * Training Started ... * \n")
os.chdir(experiment_dir)
step_size =args.steps
    
fit(train_dataset, steps=261000)  #100000 or 52 epochs (step_size/number_of_batches)=epochs


# In[13]:
###########################  RELOADING THE PRETRAINED MODEL FOR INFERENCE WITHOUT NEED FOR RETRAINING  ########################### 

#model_path='/home/mojjaf/Direct Normalization/Code/experiments/Pix2Pix_Normalization_all_phantoms_traning_20230118-141522/triple3.3_itoi/gen_18.h5'

#test_img_low='/home/mojjaf/Direct Normalization/Data/Contrast Phantom Sim/test_sim'
#test_img_full='/home/mojjaf/Direct Normalization/Data/Contrast Phantom Sim/ND_100p/Clinder_fullcount_ND.vox'
#generator= Generator2D()
#generator.load_weights(model_path)

# In[14]:


############################  Run the trained model on a few examples from the test set ########################### 
for inp, tar in val_dataset.take(15):
    generate_images(generator, inp, tar)


# In[15]:
###########################  TESTING THE MODEL ON THE VALIDATAION DATASET ########################### 

#image_val2='/home/mojjaf/Direct Normalization/Data/Test Data/LC Brain'
#image_val_target2='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/BrainPET.vox'

image_val2='/home/mojjaf/Direct Normalization/Data/Test Data/LC contrast'
image_val_target2='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/ContrastNew_fullcount.vox'


input_image_val2,real_image_val2=pairmatch_vox_files_from_directory_2D(image_val2,image_val_target2, IMG_HEIGHT, IMG_WIDTH)
datasetxVal = tf.data.Dataset.from_tensor_slices((input_image_val2))
datasetyVal = tf.data.Dataset.from_tensor_slices((real_image_val2))
val_dataset2 = tf.data.Dataset.zip((datasetxVal, datasetyVal)).batch(BATCH_SIZE)

for inp, tar in val_dataset2.take(15):
    generate_images(generator, inp, tar)


# In[16]:
###########################  INFERENCE CODE ########################### 
def denorm(yhat,dyrang):
    return dyrang*(yhat+1)/2

def generate_synthetic_sensitivity_image_2D(image_dirA,image_dirB, model, out_path):
    for image_file in (os.listdir(image_dirA)):
        #print(image_file)
        ImgA= load_cuda_vox(os.path.join(image_dirA, image_file)) #input image
        ImgB= load_cuda_vox(image_dirB) #target image file name is equal to the input image file name
        pred_vol=[]
        height=ImgA.shape[0]
        width=ImgA.shape[1]
        zlen=ImgA.shape[2]
        #print(np.shape(ImgA),np.shape(ImgB))
        for plane in range(zlen):
            ImgA_2D = ImgA[:, :, plane]
            ImgA_2D=np.expand_dims(ImgA_2D, axis=-1)
            
            ImgB_2D=ImgB[:,:,plane]
            
                       
            input_image=tf.image.resize(ImgA_2D, [IMG_HEIGHT,IMG_WIDTH],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #input_image=np.expand_dims(input_image, axis=-1)
            
            input_image_norm=normalise_per_channel(input_image)
            input_image_bt=np.expand_dims(input_image_norm, axis=0)
            
            pred_image = model(input_image_bt, training=True)
            print(np.shape(pred_image))
            pred_image=np.squeeze(pred_image, axis=0)
            plt.imshow(pred_image[...,0])
            plt.show()
           
            
            dyanamic_rang=(np.max(ImgB_2D)-np.min(ImgB_2D)) 
            pred_image=denorm(np.array(pred_image),np.array(dyanamic_rang))
           
  
           # resize image to original shape
            pred_image = cv2.resize(pred_image, (width, height), interpolation = cv2.INTER_AREA)

            pred_vol.append(pred_image)
        
        #out_path=image_dirA+'/'+'synthetic_data'
        #if not os.path.exists(out_path):
            #os.makedirs(out_path)
        img_file=out_path+'/'+'synthetic2D_'+image_file
        pred_vol=np.asarray(pred_vol)
        #print(np.shape(pred_vol))
        image=np.transpose(pred_vol, axes=[1, 2, 0])
        print(np.shape(image))
        write_cuda_vox(image, img_file, magic_number=65531, version_number=1)
        print('image saved for:',image_file)
    return 


# In[17]:



#generator.load_weights(model_path)
test_img_cyl='/home/mojjaf/Direct Normalization/Data/Test Data/LC Cylinder'
test_img_cyl_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/Clinder_FullCount.vox'
generate_synthetic_sensitivity_image_2D(test_img_cyl,test_img_cyl_target, generator,output_preds_2D)

test_img_MinRes='/home/mojjaf/Direct Normalization/Data/Test Data/LC MiniRes'
test_img_MinRes_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/MiniResolution_FullCount.vox'
generate_synthetic_sensitivity_image_2D(test_img_MinRes,test_img_MinRes_target, generator,output_preds_2D)

test_img_hoffman='/home/mojjaf/Direct Normalization/Data/Test Data/LC Hoffman'
test_img_hoffman_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/BrainHoffman_FullND.vox'
generate_synthetic_sensitivity_image_2D(test_img_hoffman,test_img_hoffman_target, generator,output_preds_2D)

test_img_contrast='/home/mojjaf/Direct Normalization/Data/Test Data/LC contrast'
test_img_contrast_target='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/ContrastNew_fullcount.vox'
generate_synthetic_sensitivity_image_2D(test_img_contrast,test_img_contrast_target, generator,output_preds_2D)


test_img_hoffmanV3='/home/mojjaf/Direct Normalization/Data/Test Data/LC HoffmanV3'
test_img_hoffman_targetV3='/home/mojjaf/Direct Normalization/Data/Training Data/Targets/BrainHoffmanV3_fullND.vox'
generate_synthetic_sensitivity_image_2D(test_img_hoffmanV3,test_img_hoffman_targetV3, generator,output_preds_2D)

#test_img_low='/home/mojjaf/Direct Normalization/Data/Cylinder Phantom/test_cylinder'
#test_img_full='/home/mojjaf/Direct Normalization/Data/Cylinder Phantom/Cylinder_ND_100p/Clinder_fullcount_ND_.vox'
#generate_synthetic_sensitivity_image(test_img_low,test_img_full, generator,output_preds_2D)

