#!/usr/bin/python

from lettersegmentationOCRTesseract import getWeights
import numpy as np
import helper
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
import sys
from loss_functions import loss_function, loss_function_IQ_Max_Similarity, loss_function_IQ_Min, loss_function_IQ_SoftMin, loss_function_IQ_Avg, loss_function_IQ_Min_Gram, loss_function_IQ_Mean_Of_TopK_Min, loss_function_IQ_Adaptive_TopK_Min, loss_function_IQ_MaxSSIM
from tensorflow_vgg import vgg19
import random
import math
from PIL import Image
import cv2
import pytesseract

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)

# Set Tesseract environment variables
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files (x86)\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

def style_layers(texture_array):
    vgg = vgg19.Vgg19()
    vgg.build(texture_array)
    return dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1,
                                3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2,
                                6: vgg.conv3_1, 7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.conv3_4, 10: vgg.pool3,
                                11: vgg.conv4_1, 12: vgg.conv4_2, 13: vgg.conv4_3, 14: vgg.conv4_4, 15: vgg.pool4,
                                16: vgg.conv5_1, 17: vgg.conv5_2, 18: vgg.conv5_3, 19: vgg.conv5_4, 20: vgg.pool5 })

def convertTensorToPill(sess, image):
     # Convert TensorFlow tensor to NumPy array
    numpy_image = sess.run(image)

    # Remove batch dimension if needed (e.g. shape: (1, H, W, C) → (H, W, C))
    if numpy_image.ndim == 4:
        numpy_image = np.squeeze(numpy_image, axis=0)

    # Convert float32 [0, 1] to uint8 [0, 255] if needed
    if numpy_image.dtype == np.float32 or numpy_image.max() <= 1.0:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(numpy_image)

    return pil_image

def compute_ocr_confidence(initial_img, recovered_img):
    custom_config = r'--oem 3 --psm 6 -l grc+lat+en'  # Specify Tesseract OCR options

    data = pytesseract.image_to_data(initial_img, output_type=pytesseract.Output.DICT, config=custom_config)
    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
    if not confidences:
        return 0.0
    initial_img_avg =  sum(confidences) / len(confidences)

    data = pytesseract.image_to_data(recovered_img, output_type=pytesseract.Output.DICT, config=custom_config)
    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
    if not confidences:
        return 0.0
    recovered_img_avg =  sum(confidences) / len(confidences)

    return recovered_img_avg / initial_img_avg

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Because we just want to compute output of convolution layers not fully connected layers,
# we can have any size of input image we want.
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

# execution_case = int(input("What do you want to do?\nGive:\n\t 0 for expanding texture\n\t 1 for shrinking texture,\n\t 2 for left tiling,\n\t 3 for right tiling,\n\t 4 for up tiling,\n\t 5 for down tiling\n\t 6 for constrained tiling\nMethod case: "))
# input_w = int(sys.argv[1])
# input_h = int(sys.argv[2])
input_folder = "./Images/epigrafes/Processed/"

test_folder = sys.argv[1]
filename = sys.argv[2]
mask_filename = sys.argv[3]
train_folder = input_folder + sys.argv[4]
exemplarInputsFilenames = []

#input_file = "./image_resources/original/" + filename
#output_path = "./image_resources/processed/"
input_file = input_folder + test_folder
output_path = "./Results/processed/"
output_file = "_processed.jpg"

# Read width and height from the specified filename
image_path = os.path.join(input_file, filename)
with Image.open(image_path) as img:
    input_w, input_h = img.size  # Get the width and height of the image

image_mask_path = os.path.join(input_file, mask_filename)
# Open the mask and get its size, convert inside the block
with Image.open(image_mask_path) as img_mask:
    input_mask_w, input_mask_h = img_mask.size  # Get the width and height of the mask image
    mask_np = np.array(img_mask).astype(np.float32) / 255.0  # Convert and normalize

# Assert that the dimensions match
assert (input_w, input_h) == (input_mask_w, input_mask_h), \
    f"Image and mask dimensions do not match: Image = ({input_w}, {input_h}), Mask = ({input_mask_w}, {input_mask_h})"

print(f"Test Image dimensions: Width = {input_mask_w}, Height = {input_mask_h}")

# Convert to tensor and add batch & channel dimensions
mask_tensor = tf.convert_to_tensor(mask_np, dtype=tf.float32)  # shape: (H, W)
mask_tensor = tf.expand_dims(mask_tensor, axis=0)  # shape: (1, H, W)
#mask_tensor = tf.expand_dims(mask_tensor, axis=-1)  # shape: (1, H, W, 1)

# Apply threshold to create binary mask
mask_tensor = tf.cast(mask_tensor > 0.5, tf.float32)
# Invert if black == corrupted
mask_tensor = 1.0 - mask_tensor

# List to store the image file paths
exemplarInputsFilenames = []

# Read all image files from the input folder
for f in os.listdir(train_folder):
    if f.endswith((".jpg", ".png", ".jpeg")):  # Adjust the file extensions as needed
        exemplarInputsFilenames.append(os.path.join(train_folder, f))

# Debug print to verify the file paths (optional)
print("Found train images:", exemplarInputsFilenames)

#image_dir = sys.argv[4]
#for filename in os.listdir(image_dir):
#    if filename.lower().endswith(('.png', '.jpg')):
#        image_path = os.path.join(image_dir, filename)
#        exemplarInputsFilenames.append(image_path)

# This will scale down the image to given width and height, save it and
# scale its value to [0-1] as the model expects the input to be between o and 1
# Lastly, it will make a tensorflow ready numpy array with [1, w, h, 3] dims

texture_array_input_epigrafh = helper.fix_img(image_path)

texture_arrays = []

style_layers_lists = []

for i in range(0, len(exemplarInputsFilenames)):
    texture_array = helper.fix_img(exemplarInputsFilenames[i])
    texture_arrays.append(texture_array)
    style_layers_list = style_layers(texture_array)
    style_layers_lists.append(style_layers_list)

print("VASILIS texture_array_input_epigrafh", texture_array_input_epigrafh.shape)

input_epigrafi_NEW = np.reshape(texture_array_input_epigrafh, (1, int(texture_array_input_epigrafh.shape[1]), int(texture_array_input_epigrafh.shape[2]), int(texture_array_input_epigrafh.shape[3])) )
input_epigrafi = tf.Variable(initial_value=input_epigrafi_NEW, name='input_epigrafi', dtype=tf.float32)

#random_ = tf.random.uniform(shape=(texture_array.shape[0], int(texture_array.shape[1]), int(texture_array.shape[2]), texture_array.shape[3]), minval=0, maxval=0.2)
#input_epigrafi = tf.Variable(initial_value=random_, name='input_epigrafi', dtype=tf.float32)

style_layers_list_input_epigrafh = style_layers(input_epigrafi)

# t = np.squeeze(texture_array)
# t *= 255.0
# t = np.clip(t, 0.0, 255.0).astype('uint8')
# imgNew = Image.fromarray(t, mode='RGB')
# imgNew.save('texure-original.jpg')
# imgNew.show()
input_ruined_image_path = image_path
intervals, weights = getWeights(input_ruined_image_path, exemplarInputsFilenames, True)
print('VASILIS LAYERS WEIGHTS ARE', weights)
print('VASILIS LAYERS INTERVALS ARE', intervals)
for i in range(len(intervals)):
    print("interval", intervals[i], "weight: ", weights[i])

# m will be array of tuples with 2 elements inside.
# 1st element will be layer no. and 2nd element will be weight assigned for the layer.
#m = [(0, float(0.025)), (2, float(0.025)), (5, float(0.25)), (10, float(0.35)), (15, float(0.35))]
#64x64, 64x64, 32x32, 16x16, 8x8

m = [(0, float(0.2)), (2, float(0.2)), (5, float(0.2)), (10, float(0.2)), (15, float(0.2))]

#m = [(0, float(1/20)), (1, float(1/20)), (2, float(1/20)), (3, float(1/20)), (4, float(1/20)), (5, float(1/20)), (6, float(1/20)), (7, float(1/20)), (8, float(1/20)), (9, float(1/20)), (10, float(1/20)), (11, float(1/20)), (12, float(1/20)), (13, float(1/20)), (14, float(1/20)), (15, float(1/20)), (16, float(1/20)), (17, float(1/20)), (18, float(1/20)), (19, float(1/20)), (20, float(1/20))]
#m = [(10, float(1/11)), (11, float(1/11)), (12, float(1/11)), (13, float(1/11)), (14, float(1/11)), (15, float(1/11)), (16, float(1/11)), (17, float(1/11)), (18, float(1/11)), (19, float(1/11)), (20, float(1/11))]
#m = [(5, float(1/16)), (6, float(1/16)), (7, float(1/16)), (8, float(1/16)), (9, float(1/16)), (10, float(1/16)), (11, float(1/16)), (12, float(1/16)), (13, float(1/16)), (14, float(1/16)), (15, float(1/16)), (16, float(1/16)), (17, float(1/16)), (18, float(1/16)), (19, float(1/16)), (20, float(1/16))]

final_noise = input_epigrafi  # Initialize final_noise with the initial input tensor
initial_image_tensor=tf.constant(input_epigrafi_NEW, dtype=tf.float32) # intial tensor but constant - cannot be changed

#total_loss, style_loss, tv_loss = loss_function_IQ_SoftMin(m, style_layers_lists, intervals = intervals, weights = weights, input_epigrafh_layers = style_layers_list_input_epigrafh, input_image_tensor=final_noise, initial_image_tensor=initial_image_tensor)
#total_loss, style_loss, tv_loss = loss_function_IQ_Mean_Of_TopK_Min(m, style_layers_lists, intervals = intervals, weights = weights, input_epigrafh_layers = style_layers_list_input_epigrafh, top_k=3, input_image_tensor=final_noise)
#total_loss, style_loss, tv_loss, top_k_count, top_min_count = loss_function_IQ_Adaptive_TopK_Min(m, style_layers_lists, intervals = intervals, weights = weights, input_epigrafh_layers = style_layers_list_input_epigrafh, top_k=3, input_image_tensor=final_noise)
#total_loss, style_loss, tv_loss = loss_function_IQ_MaxSSIM(m, style_layers_lists, intervals = intervals, weights = weights, input_epigrafh_layers = style_layers_list_input_epigrafh, input_image_tensor=final_noise)

total_loss, style_loss, tv_loss, psnr_loss, ssim_loss = loss_function_IQ_Min(m, style_layers_lists=style_layers_lists, style_layers_func = style_layers, intervals = intervals, weights = weights, input_image_tensor=final_noise, initial_image_tensor=initial_image_tensor, mask_tensor = None)
#loss = loss_function_IQ_Min_Gram(m, style_layers_lists, input_epigrafh_layers = style_layers_list_input_epigrafh)
print("VASILIS total_loss", total_loss)
print("VASILIS style_loss", style_loss)
print("VASILIS tv_loss", tv_loss)

# EPOCHS
# epochs = int(input("\nEnter EPOCHS FOR TRAINING: "))
epochs = 5000

#learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')
#optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_placeholder, beta1=0.99, epsilon=1e-1).minimize(loss)
#optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 30, 'maxcor': 20, 'ftol': 0, 'gtol': 0})
#optimizer = tfp.optimizer.lbfgs_minimize(loss, num_correction_pairs=20, tolerance=1e-8)

shouldUseLBFGSB = True

if shouldUseLBFGSB == True:
    # Create L-BFGS optimizer
    optimizer_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B', options={'maxiter': epochs, 'maxcor': 20, 'ftol': 0, 'gtol': 0})

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        start = time.time()

        loss_plot = []

        def loss_callback(loss_value, style_value, tv_value, psnr_value, ssim_value):
        
            #confidencePercentage = compute_ocr_confidence(convertTensorToPill(sess, final_noise), convertTensorToPill(sess, initial_image_tensor))

            # Compute SSIM per image (returns tensor of shape (1,))
            #ssim_value = tf.image.ssim(final_noise, initial_image_tensor, max_val=1.0)
            # If you want the scalar SSIM value
            #ssim_scalar = tf.reduce_mean(ssim_value)

            print("Epoch:", len(loss_plot), " Total Loss:", loss_value, 
                " Style Loss:", style_value, " TV Loss:", tv_value,
                "\n PSNR Loss:", psnr_value, " SSIM loss:", ssim_value,
                "\n")
                #"\n Confidence Percentage Improvement: ", confidencePercentage,
                #" SSIM:", sess.run(ssim_scalar))
                #" Sobel Clarity Loss:", sobel_loss, 
                #" Entropy Loss:", entr_loss,
                #" Binarization Contrast Loss:", bin_loss)

            loss_plot.append(loss_value)

        # Register loss callback to track loss values during optimization
        optimizer_lbfgs.minimize(sess, fetches=[total_loss, style_loss, tv_loss, psnr_loss, ssim_loss], loss_callback=loss_callback)

        final_noise = sess.run(input_epigrafi)
        mask_np = sess.run(mask_tensor)
        initial_image_np = sess.run(initial_image_tensor)
        final_noise = mask_np * final_noise + (1.0 - mask_np) * initial_image_np


        # Save loss values to a file
        loss_plot_file = "./Results/" + filename + "_" + str(epochs) + ".txt"
        with open(loss_plot_file, 'w') as f:
            for item in loss_plot:
                f.write("%s\n" % item)
else:
    # Define Adam optimizer setup
    learning_rate = 0.001  # You can lower to 0.001 for more stability
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        total_loss, var_list=[input_epigrafi]
    )

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        start = time.time()
        loss_plot = []

        for epoch in range(epochs):
            _, total_loss_val, style_loss_val, tv_loss_val, psnr_loss_val, gradient_loss_val, ssim_loss_val = sess.run(
                [optimizer, total_loss, style_loss, tv_loss, psnr_loss, gradient_loss, ssim_loss]
            )

            loss_plot.append(total_loss_val.item())  # Store scalar value

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Total Loss: {total_loss_val.item():.4f}, Style: {style_loss_val.item():.4f}, TV: {tv_loss_val.item():.4f}")
                print(f"PSNR: {psnr_loss_val.item():.4f}, Gradient: {gradient_loss_val.item():.4f}, SSIM: {ssim_loss_val.item():.4f}\n")

        final_noise = sess.run(input_epigrafi)

        # Save loss plot
        loss_plot_file = f"./Results/{filename}_{epochs}_adam.txt"
        with open(loss_plot_file, 'w') as f:
            for item in loss_plot:
                f.write(f"{item}\n")

'''with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    #print(tf.trainable_variables())
    start = time.time()

    init_noise = sess.run(input_epigrafi)
    loss_plot = []
    learning_rate =  0.001
    for i in range(epochs):
        #if i % 70000 == 0 and i != 0:
        #    learning_rate =  learning_rate / 5
        _, s_loss = sess.run([optimizer, loss], feed_dict={ learning_rate_placeholder: learning_rate })

        loss_plot.append(s_loss)
        if (i+1) % 1000 == 0:
            #print("learning_rate: ", learning_rate)
            print("Epoch: {}/{}".format(i+1, epochs), " Loss: ", s_loss)
        elif i == 0:
            print("Epoch: {}/{}".format(i+1, epochs), " Loss: ", s_loss)

    loss_plot_file = "./Results/" +filename +"_"+ str(epochs) + ".txt"
    with open(loss_plot_file, 'w') as f:
        for item in loss_plot:
            f.write("%s\n" % item)

    final_noise = sess.run(input_epigrafi)
'''

end = time.time()
print("Time to synthesize the new image: ", end - start)
print("initial texture: ", texture_array.shape)
print("final_noise: ", final_noise.shape)

epochs1 = range(1, len(loss_plot) + 1)
plt.plot(epochs1, loss_plot, 'b', label='loss function')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show(block=True)
plt.savefig("./Results/" + filename + "_" + str(epochs) +'plotForPaper.png')
# Post process will convert the output of CNN from [1, w, h, 3] to [w, h, 3].
# Then it will normalize the values to [0-1] and then convert into [0-255] range.
# Lastly, it will display image and save the image at provided loaction.

# output_directory = "./image_resources/outputs/"
output_directory = "./Results/"
initial_noise_file_name = filename + "_output_initial_epigrafi.jpg"
final_epigrafi_file_name = filename +"_"+ str(epochs) + ".jpg"

final_epigrafi_ = helper.post_process_and_display(final_noise, output_directory, final_epigrafi_file_name, texture_array_input_epigrafh, save_file=True)

print("final_epigrafi_: ", final_epigrafi_.shape)

output = final_epigrafi_

print("output: ", output.shape)

img = Image.fromarray(output, mode='RGB')
img.show()
imagename_MERGED = filename + "_" + str(epochs) + ".jpg"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
img.save(output_directory + imagename_MERGED)
