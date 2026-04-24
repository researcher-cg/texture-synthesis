from tensorflow_vgg import vgg19
import matplotlib.pyplot as plt
import numpy as np
import helper
import tf_helper
import tensorflow as tf
from scipy.misc import toimage

# Because we just want to compute output of convolution layers not fully connected layers,
# we can have any size of input image we want.

input_w = 256   # width of input image(original image will be scaled down to this width), width of generated image
input_h = 256   # height of input image(original image will be scaled down to this height), height of generated image


def loss_function(weights, texture_op, noise_layers):
    loss = tf.constant(0, dtype=tf.float32, name="Loss")

    for i in range(len(weights)):
        texture_filters = np.squeeze(texture_op[weights[i][0]], 0)
        texture_filters = np.reshape(texture_filters, newshape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
        #gram_matrix_texture = np.matmul(texture_filters.T, texture_filters)
        gram_matrix_texture = tf.matmul(tf.transpose(texture_filters), texture_filters)

        noise_filters = tf.squeeze(noise_layers[weights[i][0]], 0)
        noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))
        gram_matrix_noise = tf.matmul(tf.transpose(noise_filters), noise_filters)

        print("HERE A: ", texture_filters.shape[0])
        print("HERE B: ", texture_filters.shape[1])

        denominator = (4 * tf.square(tf.convert_to_tensor(texture_filters.shape[1], dtype=tf.float32)) * tf.convert_to_tensor(texture_filters.shape[0], dtype=tf.float32))

        loss += weights[i][1] * (tf.reduce_sum(tf.square(tf.subtract(gram_matrix_texture, gram_matrix_noise))) / tf.cast(denominator, tf.float32))

    return loss

def run_texture_synthesis(input_filename, processed_path, processed_filename, weights, eps, op_dir, initial_filename, final_filename):

    i_w = 256   # width of input image(original image will be scaled down to this width), width of generated image
    i_h = 256   # height of input image(original image will be scaled down to this height), height of generated image

    texture_array = helper.resize_and_rescale_img(input_filename, i_w, i_h, processed_path, processed_filename)
    print("TEXTURE ARRAY: ", texture_array.shape)
    # LEFT NEURAL NETWORK
    #with tf.device("/device:GPU:0"):
    print("\n------------- LEFT NEURAL NETWORK -------------\n")
    texture_outputs = tf_helper.compute_tf_output(texture_array)

    # RIGHT NEURAL NETWORK
    print("\n------------- RIGHT NEURAL NETWORK -------------\n")
    vgg_left = vgg19.Vgg19()

    random_ = tf.random_uniform(shape=texture_array.shape, minval=0, maxval=0.2)
    input_noise = tf.Variable(initial_value=random_, name='input_noise', dtype=tf.float32)
    vgg_left.build(input_noise)
    noise_layers_list = dict({0: vgg_left.conv1_1, 1: vgg_left.conv1_2, 2: vgg_left.pool1,
                              3: vgg_left.conv2_1, 4: vgg_left.conv2_2, 5: vgg_left.pool2,
                              6: vgg_left.conv3_1, 7: vgg_left.conv3_2, 8: vgg_left.conv3_3, 9: vgg_left.conv3_4, 10: vgg_left.pool3,
                              11: vgg_left.conv4_1, 12: vgg_left.conv4_2, 13: vgg_left.conv4_3, 14: vgg_left.conv4_4, 15: vgg_left.pool4,
                              16: vgg_left.conv5_1, 17: vgg_left.conv5_2, 18: vgg_left.conv5_3, 19: vgg_left.conv5_4, 20: vgg_left.pool5 })

    for i in range(len(noise_layers_list)):
        print("No. ", i, " ", noise_layers_list[i].name, "completed.")
    print("All layers' outputs have been computed sucessfully.")

    print("\n------------- LOSS -------------\n")
    # LOSS
    loss = loss_function(weights, texture_outputs, noise_layers_list)
    print("\n------------- OPTIMIZER -------------\n")
    # OPTMIZER
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    epochs = eps
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        init_noise = sess.run(input_noise)
        for i in range(epochs):
            _, s_loss = sess.run([optimizer, loss])
            if (i+1) % 5000 == 0:
                print("Epoch: {}/{}".format(i+1, epochs), " Loss: ", s_loss)
        final_noise = sess.run(input_noise)

    initial_noise = helper.post_process_and_display(init_noise, texture_array, op_dir, initial_filename, save_file=False)
    final_noise_ = helper.post_process_and_display(final_noise, texture_array, op_dir, final_filename, save_file=True)




weights = [(0, 0.000000001), (1, 0.000000001), (2, 0.000000001), (3, 0.000000001), (4, 0.000000001), (5, 0.000000001), (6, 0.000000001), (7, 0.000000001), (8, 0.000000001), (9, 0.000000001), (10, 0.000000001), (11, 0.000000001), (12, 0.000000001), (13, 0.000000001), (14, 0.000000001), (15, 0.000000001), (16, 0.000000001), (17, 0.000000001), (18, 0.000000001), (19, 0.000000001), (20, 0.000000001)]
#weights = [(0, 1), (2, 1), (5, 1), (10, 1), (15, 1)]
print("Configuration : 5 - Upto Pooling Layer 4")
input_file = "./image_resources/original/pebbles.jpg"
processed_path = "./image_resources/processed/"
processed_file = "pebbles_processed.jpg"
eps = 40000
output_dir = "./image_resources/outputs/"
noise_fn = "pebbles_noise.jpg"
final_fn = "pebbles_final.jpg"

run_texture_synthesis(input_file, processed_path, processed_file, weights, eps, output_dir, noise_fn, final_fn)

# TENSORBOARD
#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())
#writer.flush()
