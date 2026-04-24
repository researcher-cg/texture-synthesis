import numpy as np
import tensorflow as tf
import sys

def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

def min_gramians_calculation(style_layers_lists, noise_layers, m, i, mean_enabled = True):
    # ------------ NOISE FILTERS CALCULATIONS ONCE -------------
    #print("noise_filters: ", noise_layers[m[i][0]])
    noise_filters = tf.squeeze(noise_layers[m[i][0]], 0)
    print("noise_filters: ", noise_filters.shape)
    noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))
    print("noise_filters: ", noise_filters.shape)
    noise_filters_shape = noise_filters.get_shape().as_list()
    print("noise_filters_shape: ", noise_filters_shape)
    N = noise_filters_shape[1]

    if (mean_enabled):
        mean_noise = tf.reduce_mean(noise_filters)      # shape = (x, 1)
        print("noise mean: ", mean_noise.shape)
        noise_filters = tf.math.subtract(noise_filters, mean_noise)

    gram_matrix_noise = tf.matmul(tf.transpose(noise_filters), noise_filters) / noise_filters_shape[0]
    gram_matrix_noise_shape = gram_matrix_noise.get_shape().as_list()
    print("gram_matrix_noise: ", gram_matrix_noise_shape)

    gram_matrix_textures = []
    for j in range(len(style_layers_lists)):
        #print("texture_filters: ", style_layers[m[i][0]])
        texture_filters = tf.squeeze(style_layers_lists[j][m[i][0]], 0)
        print("texture_filters: ", texture_filters.shape)
        texture_filters =  tf.reshape(texture_filters, shape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
        print("texture_filters: ", texture_filters.shape)
        texture_filters_shape = texture_filters.get_shape().as_list()
        print("texture_filters_shape: ", texture_filters_shape)
        
        if (mean_enabled):
            mean = tf.reduce_mean(texture_filters)      # shape = (x, 1)
            print("mean: ", mean.shape)
            texture_filters = tf.math.subtract(texture_filters, mean)

        gram_matrix_texture = tf.matmul(tf.transpose(texture_filters), texture_filters) / texture_filters_shape[0]
        gram_matrix_texture_shape = gram_matrix_texture.get_shape().as_list()
        print("gram_matrix_texture: ", gram_matrix_texture_shape)
        gram_matrix_textures.append(gram_matrix_texture)

    min_gram_matrix_texture = gram_matrix_textures[0]
    for j in range(1, len(gram_matrix_textures)):
        min_gram_matrix_texture = tf.where(
            gram_matrix_noise - min_gram_matrix_texture > gram_matrix_noise - gram_matrix_textures[j],
            gram_matrix_textures[j],
            min_gram_matrix_texture
        )

    return min_gram_matrix_texture, gram_matrix_noise, N

def gramians_calculation(style_layers, noise_layers, m, i, mean_enabled = True):
    print("\n----------------------------------------")
    print("No. ", i, " ", style_layers[m[i][0]].name)
    print("----------------------------------------\n")

    #print("texture_filters: ", style_layers[m[i][0]])
    texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
    print("texture_filters: ", texture_filters.shape)
    texture_filters =  tf.reshape(texture_filters, shape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
    print("texture_filters: ", texture_filters.shape)
    texture_filters_shape = texture_filters.get_shape().as_list()
    print("texture_filters_shape: ", texture_filters_shape)

    #print("noise_filters: ", noise_layers[m[i][0]])
    noise_filters = tf.squeeze(noise_layers[m[i][0]], 0)
    print("noise_filters: ", noise_filters.shape)
    noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))
    print("noise_filters: ", noise_filters.shape)
    noise_filters_shape = noise_filters.get_shape().as_list()
    print("noise_filters_shape: ", noise_filters_shape)
    N = noise_filters_shape[1]

    if (mean_enabled):
        #sess = tf.compat.v1.Session()
        #with sess.as_default():
        #texture_filters_F_ = tf.reduce_mean(texture_filters, 1)
        #print("F texture_filters: ", texture_filters_F_)
        #print("\nF texture_filters: ", sess.run(texture_filters_F_))
        mean = tf.reduce_mean(texture_filters)      # shape = (x, 1)
        #print("\nF texture_filters: ", sess.run(mean))
        #print("\ntexture_filters: ", sess.run(texture_filters))
        #print("F texture_filters: ", texture_filters_F_)
        print("mean: ", mean.shape)
        texture_filters = tf.math.subtract(texture_filters, mean)
        #print("\ntexture_filters: ", sess.run(texture_filters))

        #noise_filters_F_ = tf.reduce_mean(noise_filters, 1)
        #print("F texture_filters: ", texture_filters_F_)
        #print("\nF noise_filters: ", sess.run(noise_filters_F_))
        mean_noise = tf.reduce_mean(noise_filters)      # shape = (x, 1)
        #print("\nF noise_filters: ", sess.run(mean_noise))
        #print("\nnoise_filters: ", sess.run(noise_filters))
        #print("F texture_filters: ", texture_filters_F_)
        print("noise mean: ", mean_noise.shape)
        noise_filters = tf.math.subtract(noise_filters, mean_noise)

    #print("\nnoise_filters: ", sess.run(noise_filters))

    # GRAM MATRICES -----------------------------------------------------------------------------------------
    gram_matrix_texture = tf.matmul(tf.transpose(texture_filters), texture_filters) / texture_filters_shape[0]
    gram_matrix_texture_shape = gram_matrix_texture.get_shape().as_list()
    print("gram_matrix_texture: ", gram_matrix_texture_shape)

    gram_matrix_noise = tf.matmul(tf.transpose(noise_filters), noise_filters) / noise_filters_shape[0]
    gram_matrix_noise_shape = gram_matrix_noise.get_shape().as_list()
    print("gram_matrix_noise: ", gram_matrix_noise_shape)
    #----------------------------------------------------------------------------------------------------------
    #denominator = (4 * (tf.convert_to_tensor(texture_filters.shape[1], dtype=tf.float32)**2 ) * (tf.convert_to_tensor(texture_filters.shape[0], dtype=tf.float32)**2 ))
    #denominator = (4 * (texture_filters_shape[1]**2 ) * (texture_filters_shape[0]**2))
    #denominator = (4 * (noise_filters_shape[1]**2 ) * (noise_filters_shape[0]**2))
    # MEAN SQUARE DISPLACEMENT

    #a = tf.compat.v1.math.reduce_max(style_layers[m[i][0]]) - tf.compat.v1.math.reduce_min(style_layers[m[i][0]])
    #style_loss += float(m[i][1])/2 * (1 - tf.image.ssim(tf.expand_dims(tf.expand_dims(gram_matrix_texture, 2), 0), tf.expand_dims(tf.expand_dims(gram_matrix_noise, 2), 0), max_val=a, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)[0])

    #loss += m[i][1] * (tf.reduce_sum(tf.square(tf.subtract(gram_matrix_texture, gram_matrix_union_halfslices))) / tf.cast(denominator, tf.float32))
    return gram_matrix_texture, gram_matrix_noise, N

def gramians_similarities_calculation(style_layers, noise_layers, m, i, mean_enabled = False):
    print("\n----------------------------------------")
    print("No. ", i, " ", style_layers[m[i][0]].name)
    print("----------------------------------------\n")

    #print("texture_filters: ", style_layers[m[i][0]])
    texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
    print("texture_filters: ", texture_filters.shape)

    texture_filters = tf.reshape(texture_filters, shape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
    print("texture_filters: ", texture_filters.shape)
    texture_filters_shape = texture_filters.get_shape().as_list()
    print("texture_filters_shape: ", texture_filters_shape)

    #print("noise_filters: ", noise_layers[m[i][0]])
    noise_filters = tf.squeeze(noise_layers[m[i][0]], 0)
    print("noise_filters: ", noise_filters.shape)
    noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))
    print("noise_filters: ", noise_filters.shape)
    noise_filters_shape = noise_filters.get_shape().as_list()
    print("noise_filters_shape: ", noise_filters_shape)
    N = noise_filters_shape[1]

    if (mean_enabled):
        #sess = tf.compat.v1.Session()
        #with sess.as_default():
        #texture_filters_F_ = tf.reduce_mean(texture_filters, 1)
        #print("F texture_filters: ", texture_filters_F_)
        #print("\nF texture_filters: ", sess.run(texture_filters_F_))
        mean = tf.reduce_mean(texture_filters)      # shape = (x, 1)
        #print("\nF texture_filters: ", sess.run(mean))
        #print("\ntexture_filters: ", sess.run(texture_filters))
        #print("F texture_filters: ", texture_filters_F_)
        print("mean: ", mean.shape)
        texture_filters = tf.math.subtract(texture_filters, mean)
        #print("\ntexture_filters: ", sess.run(texture_filters))

        #noise_filters_F_ = tf.reduce_mean(noise_filters, 1)
        #print("F texture_filters: ", texture_filters_F_)
        #print("\nF noise_filters: ", sess.run(noise_filters_F_))
        mean_noise = tf.reduce_mean(noise_filters)      # shape = (x, 1)
        #print("\nF noise_filters: ", sess.run(mean_noise))
        #print("\nnoise_filters: ", sess.run(noise_filters))
        #print("F texture_filters: ", texture_filters_F_)
        print("noise mean: ", mean_noise.shape)
        noise_filters = tf.math.subtract(noise_filters, mean_noise)

    #print("\nnoise_filters: ", sess.run(noise_filters))

    # GRAM MATRICES -----------------------------------------------------------------------------------------
    gram_matrix_texture = tf.matmul(tf.transpose(texture_filters), noise_filters) / texture_filters_shape[0]
    gram_matrix_texture_shape = gram_matrix_texture.get_shape().as_list()
    print("gram_matrix_texture: ", gram_matrix_texture_shape)

    return gram_matrix_texture, N

def filters_calculation(style_layers, epigrafi_layers, m, i):
    print("\n----------------------------------------")
    print("No. ", i, " ", style_layers[m[i][0]].name)
    print("----------------------------------------\n")

    #print("texture_filters: ", style_layers[m[i][0]])
    texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
    print("texture_filters: ", texture_filters.shape)

    #print("epigrafi_filters: ", epigrafi_layers[m[i][0]])
    epigrafi_filters = tf.squeeze(epigrafi_layers[m[i][0]], 0)
    print("epigrafi_filters: ", epigrafi_filters.shape)

    # SIMPLE FILTER MATRICES -----------------------------------------------------------------------------------------
    filters_matrix_texture = tf.Variable(texture_filters, name="filters_matrix_texture"+str(i))
    filters_matrix_texture_shape = filters_matrix_texture.get_shape().as_list()
    print("filters_matrix_texture: ", filters_matrix_texture_shape)

    filters_matrix_epigrafi = tf.Variable(epigrafi_filters, name="filters_matrix_epigrafes"+str(i))
    filters_matrix_epigrafi_shape = filters_matrix_epigrafi.get_shape().as_list()
    print("filters_matrix_epigrafi: ", filters_matrix_epigrafi_shape)
    N = filters_matrix_epigrafi_shape[2]

    return filters_matrix_texture, filters_matrix_epigrafi, N

def mean_square_displacement(gram_matrix_texture, gram_matrix_noise, N, weight):
    return float(weight)/4 * tf.reduce_sum(tf.square(tf.subtract(gram_matrix_texture, gram_matrix_noise))) / (N**2)

def root_mean_square_error(gram_matrix_texture, gram_matrix_noise, N, weight):
    squared_diff = tf.square(tf.subtract(gram_matrix_texture, gram_matrix_noise))
    mse = float(weight) * tf.reduce_sum(squared_diff) / N
    rmse = tf.sqrt(mse)
    return mse, rmse

def ssim(texture, noise, filterSize):
    return tf.image.ssim(texture, noise, max_val=1.0, filter_size=filterSize, filter_sigma=1.5, k1=0.01, k2=0.03)

def style_loss_per_layer(style_layers, noise_layers, m, i, style_layers_GaussianPyramid = None):
    '''texture_filters = np.squeeze(style_layers[m[i][0]], 0)
    print("texture_filters: ", texture_filters.shape)
    texture_filters = np.reshape(texture_filters, newshape=(texture_filters.shape[0] * texture_filters.shape[1], texture_filters.shape[2]))
    print("texture_filters: ", texture_filters.shape)
    gram_matrix_texture = np.matmul(texture_filters.T, texture_filters)

    noise_filters_array = np.squeeze(noise_layers[m[i][0]], 0)
    print("noise_filters_array: ", noise_filters_array.shape)'''
    # Main Calculation
    gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(style_layers, noise_layers, m, i)
    sum = 0
    if (style_layers_GaussianPyramid == None):
        # gram_matrix_texture_shape = gram_matrix_texture.get_shape().as_list()
        # ssim1 = tf.reshape(gram_matrix_texture, [gram_matrix_texture_shape[0], gram_matrix_texture_shape[1], 1])
        # gram_matrix_noise_shape = gram_matrix_noise.get_shape().as_list()
        # ssim2 = tf.reshape(gram_matrix_noise, [gram_matrix_noise_shape[0], gram_matrix_noise_shape[1], 1])
        
        # texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
        # noise_filters = tf.squeeze(noise_layers[m[i][0]], 0)

        # print("noise_filters: ", noise_filters.shape)
        # noise_filters = tf.reshape(noise_filters, shape=(noise_filters.shape[0] * noise_filters.shape[1], noise_filters.shape[2]))

        sum = ( mean_square_displacement( gram_matrix_texture, gram_matrix_noise, N, m[i][1] ) )
    else:
        print("len(style_layers_GaussianPyramid): ", len(style_layers_GaussianPyramid))
        # Gaussian Pyramid
        for gaussian_layer in range(1, len(style_layers_GaussianPyramid)):
            gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(style_layers_GaussianPyramid[gaussian_layer], noise_layers, m, i)
            sum += mean_square_displacement( gram_matrix_texture, gram_matrix_noise, N, m[i][1] )
            sum = sum/len(style_layers_GaussianPyramid)
    return sum


def loss_function(execution_case, style_transfer, m, style_layers, noise_layers, left_tile_style_layers = None, noise_layers_left = None,  content_layers = None, cm = None, noise_layers_content = None, style_original_list = None, original_right_Tile_style_layers = None, original_down_Tile_style_layers = None):
    style_loss = tf.constant(0, dtype=tf.float32, name="Loss")

    content_loss = tf.constant(0, dtype=tf.float32, name="Content_Loss")

    for i in range(len(m)):
        #seamless_n = style_loss_per_layer(seamless_original, seamless_noise, m, i)
        #seamless_o = style_loss_per_layer(style_layers, noise_layers, m, i)
        #style_loss += ( seamless_n + seamless_o ) + ( seamless_n / seamless_o )

        if(execution_case == 6):
            style_loss_of_produced_tiles_maintain_seamless = style_loss_per_layer(style_layers, noise_layers, m, i) + style_loss_per_layer(left_tile_style_layers, noise_layers_left, m, i)
            #style_loss_of_produced_tiles_synthesizeOtherPlaces = style_loss_per_layer(original_right_Tile_style_layers, noise_layers, m, i)/2 + style_loss_per_layer(original_down_Tile_style_layers, noise_layers_left, m, i)/2

            #style_loss_of_noise_tiles = style_loss_per_layer(noise_layers, noise_layers_left, m, i)
            style_loss += style_loss_of_produced_tiles_maintain_seamless#/2 + style_loss_of_noise_tiles/2
        else:
             style_loss += style_loss_per_layer(style_layers, noise_layers, m, i)
        print("----------------------------------------\n")

    if(style_transfer == 1):

        for i in range(len(cm)):
            if (execution_case >= 2 and execution_case < 6):
                noise_filters = noise_layers_content[cm[i][0]]
                tf.print(noise_filters, output_stream=sys.stdout)
                content_filters = content_layers[cm[i][0]]
                tf.print(content_filters, output_stream=sys.stdout)
            else:
                noise_filters = noise_layers[cm[i][0]]
                tf.print(noise_filters, output_stream=sys.stdout)
                content_filters = content_layers[cm[i][0]]
                tf.print(content_filters, output_stream=sys.stdout)
            print("content_filters: ", content_filters)
            print("noise_filters: ", noise_filters)

            #mean = tf.reduce_mean(content_filters)      # shape = (x, 1)
            #print("mean: ", mean.shape)
            #content_filters = tf.math.subtract(content_filters, mean)
            
            #mean_noise = tf.reduce_mean(noise_filters)      # shape = (x, 1)
            #print("noise mean: ", mean_noise.shape)
            #noise_filters = tf.math.subtract(noise_filters, mean_noise)
            
            content_loss += tf.reduce_sum(tf.square(tf.subtract(noise_filters, content_filters)))/2
            #content_loss += tf.image.ssim(noise_filters, content_filters, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)[0]

        b = float( 1 / (1 + (8*1e-6)) )
        a = float(1 - b)
        loss = a*content_loss + b*style_loss

        #loss += total_variation_weight*total_variation_loss(texture_array2)
    else:
        loss = style_loss
        #loss += total_variation_weight*total_variation_loss(texture_array)

    return loss

def style_loss_per_layer_IQ_Min(style_layers_lists, input_epigrafh_layers, m, i):
    minMSDisplacement = tf.constant(sys.maxsize, dtype=tf.float32, name="MinLossOfLayer"+str(i))

    for j in range(len(style_layers_lists)):
        gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(style_layers_lists[j], input_epigrafh_layers, m, i)
        MSDisplacement = mean_square_displacement(gram_matrix_texture, gram_matrix_noise, N, m[i][1])

        #MSE, RMSE = root_mean_square_error(gram_matrix_texture, gram_matrix_noise, N, m[i][1])
        #MSDisplacement = RMSE

        condition = tf.less(minMSDisplacement, MSDisplacement)
        minMSDisplacement = tf.cond(condition, lambda: minMSDisplacement, lambda: MSDisplacement)
        print('VASILIS minMSDisplacement', minMSDisplacement)

        '''with tf.Session() as sess:
            p = tf.print(minMSDisplacement)
            sess.run(p)'''

    return minMSDisplacement

def sobel_filter_2d(image):
    # image shape: (B, H, W, C)
    sobel_x = tf.constant([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=tf.float32)
    sobel_y = tf.constant([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=tf.float32)
    
    sobel_x = sobel_x[:, :, tf.newaxis, tf.newaxis]  # (3, 3, 1, 1)
    sobel_y = sobel_y[:, :, tf.newaxis, tf.newaxis]

    C = image.shape[-1]
    sobel_x = tf.tile(sobel_x, [1, 1, C, 1])  # (3, 3, C, 1)
    sobel_y = tf.tile(sobel_y, [1, 1, C, 1])  # (3, 3, C, 1)

    gx = tf.nn.depthwise_conv2d(image, sobel_x, strides=[1,1,1,1], padding='SAME')
    gy = tf.nn.depthwise_conv2d(image, sobel_y, strides=[1,1,1,1], padding='SAME')

    grad = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)

    return grad

def edge_aware_gramOLD(features):
    edge_feats = sobel_filter_2d(features)
    edge_feats = tf.reshape(edge_feats, [edge_feats.shape[0], -1])
    N = tf.shape(edge_feats)[1]  # number of spatial positions (pixels)
    N = tf.cast(N, tf.float32) 

    gram = tf.matmul(edge_feats, edge_feats, transpose_b=True) / tf.cast(N, tf.float32)
def edge_aware_gram(features):
    edge_feats = sobel_filter_2d(features)  # shape: (B, H, W, C)
    shape = tf.shape(edge_feats)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    N = H * W

    edge_feats = tf.reshape(edge_feats, [B, N, C])  # flatten spatial dims

    mean_feats = tf.reduce_mean(edge_feats, axis=1, keepdims=True)  # mean per channel
    centered_feats = edge_feats - mean_feats

    N_float = tf.cast(N, tf.float32)
    gram = tf.matmul(centered_feats, centered_feats, transpose_a=True) / N_float  # (B, C, C)

    return gram, N_float

def style_loss_per_layer_edgeaware_Mean(style_layers_lists, input_epigrafh_layers, m, i):
    msd_list = []
    gram_matrix_input, N = edge_aware_gram(input_epigrafh_layers[m[i][0]])

    for j in range(len(style_layers_lists)):
        gram_matrix_exemplar, _ = edge_aware_gram(style_layers_lists[j][m[i][0]])

        MSDisplacement = mean_square_displacement(gram_matrix_exemplar, gram_matrix_input, N, m[i][1])
        msd_list.append(MSDisplacement)

    mean_MSD = tf.reduce_mean(msd_list)
    return m[i][1] * mean_MSD

def content_loss_per_layer_min(style_layers_lists, input_epigrafh_layers, m, i):
    layer_name = m[i][0]
    weight = m[i][1]

    input_layer = input_epigrafh_layers[layer_name]
    
    min_loss = tf.constant(sys.maxsize, dtype=tf.float32)

    for j in range(len(style_layers_lists)):
        exemplar_layer = style_layers_lists[j][layer_name]

        # Use L2 (MSE) or L1 depending on your preference
        layer_loss = tf.reduce_mean(tf.square(input_layer - exemplar_layer))

        min_loss = tf.minimum(min_loss, layer_loss)

    # Apply the layer-specific weight
    return weight * min_loss

def content_loss_per_layer_min_cosine(style_layers_lists, input_epigrafh_layers, m, i, pooling=True, epsilon=1e-8):
    layer_name = m[i][0]
    weight = m[i][1]

    input_layer = input_epigrafh_layers[layer_name]
    
    min_loss = tf.constant(sys.maxsize, dtype=tf.float32)

    for j in range(len(style_layers_lists)):
        exemplar_layer = style_layers_lists[j][layer_name]

        input_feat = input_layer
        exemplar_feat = exemplar_layer

        # Optional average pooling to align structure or reduce dimensions
        if pooling:
            input_feat = tf.nn.avg_pool(input_feat, ksize=2, strides=2, padding='SAME')
            exemplar_feat = tf.nn.avg_pool(exemplar_feat, ksize=2, strides=2, padding='SAME')

        # Flatten
        input_flat = tf.reshape(input_feat, [1, -1])
        exemplar_flat = tf.reshape(exemplar_feat, [1, -1])

        # Normalize
        input_norm = tf.nn.l2_normalize(input_flat, axis=1, epsilon=epsilon)
        exemplar_norm = tf.nn.l2_normalize(exemplar_flat, axis=1, epsilon=epsilon)

        # Cosine similarity
        cosine_sim = tf.reduce_sum(input_norm * exemplar_norm, axis=1)
        cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)

        # Convert similarity to loss: lower is better
        cosine_loss = 1.0 - cosine_sim[0]

        # Keep minimum over all exemplars
        min_loss = tf.minimum(min_loss, cosine_loss)

    return weight * min_loss

def gram_linear(X):
    """Compute linear kernel (Gram matrix) for CKA."""
    return tf.matmul(X, X, transpose_b=True)

def center_gram(K):
    """Center Gram matrix K."""
    n = tf.shape(K)[0]
    one_n = tf.ones((n, n), dtype=K.dtype) / tf.cast(n, K.dtype)
    K_centered = K - tf.matmul(one_n, K) - tf.matmul(K, one_n) + tf.matmul(one_n, tf.matmul(K, one_n))
    return K_centered

def hsic(K, L):
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) between centered Gram matrices K and L."""
    n = tf.cast(tf.shape(K)[0], K.dtype)
    return tf.reduce_sum(K * L) / (n - 1)**2

def cka_linear(X, Y):
    """Compute linear CKA between two feature matrices X and Y."""
    # Flatten spatial dims to one dimension if needed
    X_flat = tf.reshape(X, (tf.shape(X)[0], -1))  # [batch, features] flatten height*width*channels if present
    Y_flat = tf.reshape(Y, (tf.shape(Y)[0], -1))
    
    K = gram_linear(X_flat)
    L = gram_linear(Y_flat)
    
    Kc = center_gram(K)
    Lc = center_gram(L)
    
    hsic_xy = hsic(Kc, Lc)
    hsic_xx = hsic(Kc, Kc)
    hsic_yy = hsic(Lc, Lc)
    
    return hsic_xy / tf.sqrt(hsic_xx * hsic_yy + 1e-12)  # add epsilon to avoid div0

def content_loss_per_layer_min_cka(style_layers_lists, input_epigrafh_layers, m, i, eps=1e-8):
    layer_name = m[i][0]
    weight = m[i][1]

    input_layer = input_epigrafh_layers[layer_name]
    input_layer = tf.reshape(input_layer, [input_layer.shape[0], -1])  # Flatten spatial
    input_layer = tf.nn.l2_normalize(input_layer, axis=1)  # L2 normalize

    def center_gram(g):
        g_mean = tf.reduce_mean(g, axis=0, keepdims=True)
        g_centered = g - g_mean - tf.transpose(g_mean) + tf.reduce_mean(g)
        return g_centered

    input_gram = tf.matmul(input_layer, input_layer, transpose_b=True)
    input_gram = center_gram(input_gram)

    min_loss = tf.constant(float('inf'), dtype=tf.float32)

    for j in range(len(style_layers_lists)):
        exemplar_layer = style_layers_lists[j][layer_name]
        exemplar_layer = tf.reshape(exemplar_layer, [exemplar_layer.shape[0], -1])
        exemplar_layer = tf.nn.l2_normalize(exemplar_layer, axis=1)

        exemplar_gram = tf.matmul(exemplar_layer, exemplar_layer, transpose_b=True)
        exemplar_gram = center_gram(exemplar_gram)

        hsic = tf.reduce_sum(input_gram * exemplar_gram)
        hsic_xx = tf.reduce_sum(input_gram * input_gram)
        hsic_yy = tf.reduce_sum(exemplar_gram * exemplar_gram)

        denom = tf.sqrt(hsic_xx * hsic_yy) + eps
        cka = hsic / denom

        # Loss to minimize: 1 - CKA
        layer_loss = 1.0 - cka

        min_loss = tf.minimum(min_loss, layer_loss)

    return weight * min_loss

def content_loss_per_layer_mean_cka(style_layers_lists, input_epigrafh_layers, m, i, eps=1e-8):
    layer_name = m[i][0]
    weight = m[i][1]

    input_layer = input_epigrafh_layers[layer_name]
    input_layer = tf.reshape(input_layer, [input_layer.shape[0], -1])  # Flatten spatial
    input_layer = tf.nn.l2_normalize(input_layer, axis=1)  # L2 normalize

    def center_gram(g):
        g_mean = tf.reduce_mean(g, axis=0, keepdims=True)
        g_centered = g - g_mean - tf.transpose(g_mean) + tf.reduce_mean(g)
        return g_centered

    input_gram = tf.matmul(input_layer, input_layer, transpose_b=True)
    input_gram = center_gram(input_gram)

    total_layer_loss = 0.0
    num_exemplars = len(style_layers_lists)

    for j in range(num_exemplars):
        exemplar_layer = style_layers_lists[j][layer_name]
        exemplar_layer = tf.reshape(exemplar_layer, [exemplar_layer.shape[0], -1])
        exemplar_layer = tf.nn.l2_normalize(exemplar_layer, axis=1)

        exemplar_gram = tf.matmul(exemplar_layer, exemplar_layer, transpose_b=True)
        exemplar_gram = center_gram(exemplar_gram)

        hsic = tf.reduce_sum(input_gram * exemplar_gram)
        hsic_xx = tf.reduce_sum(input_gram * input_gram)
        hsic_yy = tf.reduce_sum(exemplar_gram * exemplar_gram)

        denom = tf.sqrt(hsic_xx * hsic_yy) + eps
        cka = hsic / denom

        layer_loss = 1.0 - cka  # We want to minimize 1 - CKA
        total_layer_loss += layer_loss

    mean_loss = total_layer_loss / num_exemplars
    return weight * mean_loss

def style_loss_per_layer_IQ_TopKMean(style_layers_lists, input_epigrafh_layers, m, i, top_k=3):
    msd_list = []

    for j in range(len(style_layers_lists)):
        # Compute Gram matrices and MSD
        gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(style_layers_lists[j], input_epigrafh_layers, m, i)
        MSDisplacement = mean_square_displacement(gram_matrix_texture, gram_matrix_noise, N, m[i][1])
        msd_list.append(MSDisplacement)

    # Stack and sort to get Top-K smallest MSDs
    msd_tensor = tf.stack(msd_list)
    topk_msd = tf.sort(msd_tensor)[:top_k]

    # Return the average of Top-K
    return tf.reduce_mean(topk_msd)

def adaptive_style_loss_IQ(style_layers_lists, input_epigrafh_layers, m, i, k=5, var_threshold=1e-2):
    msd_list = []

    for j in range(len(style_layers_lists)):
        gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(
            style_layers_lists[j], input_epigrafh_layers, m, i)
        MSDisplacement = mean_square_displacement(
            gram_matrix_texture, gram_matrix_noise, N, m[i][1])
        msd_list.append(MSDisplacement)

    losses_tensor = tf.stack(msd_list)
    variance = tf.math.reduce_variance(losses_tensor)

    def use_min():
        return tf.reduce_min(losses_tensor), tf.constant(0), tf.constant(1)

    def use_top_k():
        top_k_vals = tf.sort(losses_tensor)[:k]
        return tf.reduce_mean(top_k_vals), tf.constant(1), tf.constant(0)

    loss, top_k_count, top_min_count = tf.cond(
        tf.less(variance, var_threshold),
        use_min,
        use_top_k
    )

    return loss, top_k_count, top_min_count, variance

def sobel_strength_loss(image):
    # returns a positive measure of edge strength
    sobel = tf.image.sobel_edges(image)
    gx = sobel[..., 0]
    gy = sobel[..., 1]
    mag = tf.sqrt(tf.square(gx) + tf.square(gy))
    return tf.reduce_mean(mag)      # ≥ 0

def entropy_loss(image):
    # returns a positive entropy measure
    img = tf.clip_by_value(image, 1e-8, 1.0)
    hist = tf.histogram_fixed_width(img, [0.0, 1.0], nbins=256)
    p = hist / tf.reduce_sum(hist)
    return -tf.reduce_sum(p * tf.math.log(p + 1e-8))  # ≥ 0

def binarization_contrast_loss(image):
    # peaks at 0.25 when image==0.5, else smaller
    return tf.reduce_mean(image * (1.0 - image))  # ≥ 0

def style_loss_per_layer_IQ_SoftMin(style_layers_lists, input_epigrafh_layers, m, i, temperature=0.1):
    losses = []
    for j in range(len(style_layers_lists)):
        gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(
            style_layers_lists[j], input_epigrafh_layers, m, i
        )
        MSDisplacement = mean_square_displacement(gram_matrix_texture, gram_matrix_noise, N, m[i][1])
        losses.append(MSDisplacement)

    # Stack and compute softmin
    losses_tensor = tf.stack(losses)
    weights = tf.nn.softmax(-losses_tensor / temperature)
    return tf.reduce_sum(weights * losses_tensor)

def style_loss_per_layer_IQ_Min_Gram(style_layers_lists, input_epigrafh_layers, m, i):
    print('\nVASILIS ---------- LAYER i: ', str(i), '----------')

    min_gram_matrix_texture, gram_matrix_noise, N = min_gramians_calculation(style_layers_lists, input_epigrafh_layers, m, i)
    MSDisplacement = mean_square_displacement(min_gram_matrix_texture, gram_matrix_noise, N, m[i][1])

    return MSDisplacement

def normalize_gram(G):
    min_val = tf.reduce_min(G)
    max_val = tf.reduce_max(G)
    return (G - min_val) / (max_val - min_val + 1e-8)

def style_loss_per_layer_IQ_MaxSSIM(style_layers_lists, input_epigrafh_layers, m, i):
    maxSSIM = tf.constant(-1.0, dtype=tf.float32, name="MaxSSIMLossOfLayer"+str(i))  # SSIM range is [-1, 1]

    for j in range(len(style_layers_lists)):
        gram_matrix_texture, gram_matrix_epigrafi, N = gramians_calculation(
            style_layers_lists[j], input_epigrafh_layers, m, i
        )

        # Normalize the Gram matrices
        gram_matrix_texture = normalize_gram(gram_matrix_texture)
        gram_matrix_epigrafi = normalize_gram(gram_matrix_epigrafi)

        # Add channel dimension for SSIM
        gram_matrix_texture = tf.expand_dims(gram_matrix_texture, axis=-1)
        gram_matrix_epigrafi = tf.expand_dims(gram_matrix_epigrafi, axis=-1)

        # Compute SSIM
        ssim_score = tf.image.ssim(gram_matrix_texture, gram_matrix_epigrafi, max_val=1.0)

        # Ensure scalar
        ssim_score = tf.reduce_mean(ssim_score)
        maxSSIM = tf.maximum(maxSSIM, ssim_score)

        tf.print('SSIM for j=', j, ':', ssim_score, 'Max so far:', maxSSIM)

    # Final loss is (1 - maxSSIM) * weight
    return maxSSIM * m[i][1]

def style_loss_per_layer_IQ_TopKMaxSSIM(style_layers_lists, input_epigrafh_layers, m, i, top_k):
    ssim_scores = []

    for j in range(len(style_layers_lists)):
        gram_matrix_texture, gram_matrix_epigrafi, N = gramians_calculation(
            style_layers_lists[j], input_epigrafh_layers, m, i
        )

        # Normalize Gram matrices
        gram_matrix_texture = normalize_gram(gram_matrix_texture)
        gram_matrix_epigrafi = normalize_gram(gram_matrix_epigrafi)

        # Add channel dimension
        gram_matrix_texture = tf.expand_dims(gram_matrix_texture, axis=-1)
        gram_matrix_epigrafi = tf.expand_dims(gram_matrix_epigrafi, axis=-1)

        # Compute SSIM and reduce to scalar
        ssim_score = tf.image.ssim(gram_matrix_texture, gram_matrix_epigrafi, max_val=1.0)
        ssim_score = tf.reduce_mean(ssim_score)
        ssim_scores.append(ssim_score)

    # Convert list to tensor
    ssim_scores_tensor = tf.stack(ssim_scores)

    # Get Top-K SSIM values
    top_k_values, _ = tf.math.top_k(ssim_scores_tensor, k=top_k, sorted=False)

    # Mean of Top-K SSIM values
    top_k_mean_ssim = tf.reduce_mean(top_k_values)

    tf.print("Top-K SSIM values:", top_k_values, "Top-K SSIM mean:", top_k_mean_ssim)

    # Final weighted loss
    return top_k_mean_ssim * m[i][1]


def style_loss_per_layer_IQ_Max_Similarity(style_layers_lists, input_epigrafh_layers, m, i):
    maxMeanLoss = tf.constant(-sys.maxsize, dtype=tf.float32, name="MaxLossOfLayer"+str(i))
    
    for j in range(len(style_layers_lists)):
        gram_matrix_texture, N = gramians_similarities_calculation(style_layers_lists[j], input_epigrafh_layers, m, i)
        newMean = tf.reduce_mean(gram_matrix_texture)
        maxMeanLoss = tf.maximum(maxMeanLoss, newMean)

        print('VASILIS maxMeanLoss', maxMeanLoss)

    return maxMeanLoss

def style_loss_per_layer_IQ_Avg(style_layers_lists, input_epigrafh_layers, m, i):
    sum = 0

    for j in range(len(style_layers_lists)):
        gram_matrix_texture, gram_matrix_noise, N = gramians_calculation(style_layers_lists[j], input_epigrafh_layers, m, i)
        MSDisplacement = mean_square_displacement( gram_matrix_texture, gram_matrix_noise, N, m[i][1] )
        #MSE, RMSE = root_mean_square_error(gram_matrix_texture, gram_matrix_noise, N, m[i][1])
        #MSDisplacement = RMSE
        sum = sum + MSDisplacement
   
    return sum/len(style_layers_lists)

def calculate_weight(width, intervals, weights):
    for i in range(len(intervals)):
        if (width <= intervals[i][1] and width >= intervals[i][0]):
            return weights[i]
    return 1

def normalize_weights(m):
    # Extract weights from the tuples
    weights = [weight for _, weight in m]
    
    # Calculate the sum of all weights
    total_weight = sum(weights)
    
    # Normalize the weights by dividing each by the total weight
    normalized_weights = [weight / total_weight for weight in weights]
    
    # Replace the old weights with the normalized weights in the list of tuples
    normalized_m = [(index, normalized_weights[i]) for i, (index, _) in enumerate(m)]
    
    return normalized_m

def psnr_loss(y_true, y_pred, max_val=1.0, eps=1e-8):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = 20.0 * tf.math.log(max_val / tf.sqrt(mse + eps)) / tf.math.log(10.0)
    return -psnr  # για ελαχιστοποίηση

def gradient_loss(y_true, y_pred):
    sobel_x = tf.image.sobel_edges(y_pred)[..., 0]
    sobel_y = tf.image.sobel_edges(y_pred)[..., 1]
    sobel_pred = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))

    sobel_x_true = tf.image.sobel_edges(y_true)[..., 0]
    sobel_y_true = tf.image.sobel_edges(y_true)[..., 1]
    sobel_true = tf.sqrt(tf.square(sobel_x_true) + tf.square(sobel_y_true))

    return tf.reduce_mean(tf.abs(sobel_pred - sobel_true))

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def total_variation_loss(tensor_image, mask_tensor):
    """
    Compute total variation loss inside masked areas.
    tensor_image: shape [1, H, W, C]
    mask_tensor: shape [1, H, W, 1], with 1.0 in corrupted areas
    """
    # Vertical variation
    dy = tf.square(tensor_image[:, 1:, :, :] - tensor_image[:, :-1, :, :])
    mask_y = mask_tensor[:, 1:, :, :] * mask_tensor[:, :-1, :, :]
    dy *= mask_y
    dy_loss = tf.reduce_sum(dy)
    dy_norm = tf.reduce_sum(mask_y) + 1e-8

    # Horizontal variation
    dx = tf.square(tensor_image[:, :, 1:, :] - tensor_image[:, :, :-1, :])
    mask_x = mask_tensor[:, :, 1:, :] * mask_tensor[:, :, :-1, :]
    dx *= mask_x
    dx_loss = tf.reduce_sum(dx)
    dx_norm = tf.reduce_sum(mask_x) + 1e-8

    return (dy_loss / dy_norm + dx_loss / dx_norm) / 2.0

def create_gaussian_kernel(filter_size=5, sigma=1.0):
    """Generates a 2D Gaussian kernel."""
    ax = np.arange(-filter_size // 2 + 1., filter_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    kernel = kernel.astype(np.float32)
    kernel = kernel[:, :, np.newaxis, np.newaxis]  # Shape: [H, W, in_channels, channel_multiplier]
    return kernel

def gaussian_blur(mask_tensor, filter_size=5, sigma=1.0):
    """Apply Gaussian blur using depthwise convolution in TF 1.x."""
    input_channels = mask_tensor.get_shape().as_list()[-1]
    kernel = create_gaussian_kernel(filter_size, sigma)
    kernel = np.repeat(kernel, input_channels, axis=2)  # Match input channels
    kernel_tensor = tf.constant(kernel, dtype=tf.float32)

    blurred = tf.nn.depthwise_conv2d(
        mask_tensor,
        kernel_tensor,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    return blurred

def loss_function_IQ_Min(m, style_layers_func, style_layers_lists, intervals, weights, input_image_tensor=None, initial_image_tensor=None, mask_tensor=None):
    # Ensure that the uncorrupted area is kept clean
    if mask_tensor is not None and initial_image_tensor is not None:
        # Assuming mask_tensor is a float32 tensor with shape [batch_size, height, width, 1]
        #blurred_mask = gaussian_blur(mask_tensor, filter_size=5, sigma=1.0)
        input_image_tensor = mask_tensor * input_image_tensor + (1 - mask_tensor) * initial_image_tensor
    
    input_epigrafh_layers = style_layers_func(input_image_tensor)

    style_loss = tf.Variable(0.0, dtype=tf.float32, name="StyleLoss")  # Define style_loss as a variable
    content_loss = tf.Variable(0.0, dtype=tf.float32, name="ContentLoss")  # Define style_loss as a variable
    #edge_style_loss = tf.Variable(0.0, dtype=tf.float32, name="EdgeStyleLoss")  # Define style_loss as a variable

    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    modified_m = []  # New list to store modified tuples

    filters_widths_shapes = []
    style_layers = style_layers_lists[0]
    for i in range(len(m)):
        texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
        print("texture_filters: ", texture_filters.shape)
        texture_filters_shape = texture_filters.get_shape().as_list()
        width = texture_filters_shape[0]
        filters_widths_shapes.append(width)
        weight = calculate_weight(width, intervals, weights)
        # Create a new tuple with the updated weight
        new_tuple = (m[i][0], weight)
        modified_m.append(new_tuple)

    print('VASILIS filters WIDTH shapes:', filters_widths_shapes)
    print('VASILIS weights:', filters_widths_shapes)

    normalised_m = normalize_weights(modified_m)

    # Convert modified_m back to tuple-like list of tuples
    m_new = tuple(normalised_m)
    print("Modified m as tuples:", m_new)

    for i in range(len(modified_m)):
        style_loss = style_loss + style_loss_per_layer_IQ_Min(style_layers_lists, input_epigrafh_layers, m_new, i)
        #edge_style_loss = edge_style_loss + style_loss_per_layer_edgeaware_Mean(style_layers_lists, input_epigrafh_layers, m_new, i)
        #content_loss = content_loss + content_loss_per_layer_mean_cka(style_layers_lists, input_epigrafh_layers, m_new, i)

    tv_loss = tf.Variable(0.0, dtype=tf.float32, name="TVLoss")  # Define style_loss as a variable
    if (input_image_tensor != None):
        tv_loss = tf.image.total_variation(input_image_tensor)
        #tv_loss = total_variation_loss(input_image_tensor, mask_tensor)
    
    if (input_image_tensor != None and initial_image_tensor != None):
        psnr_l = psnr_loss(initial_image_tensor, input_image_tensor)
        #gradient_l = gradient_loss(initial_image_tensor, input_image_tensor)
        ssim_l = ssim_loss(initial_image_tensor, input_image_tensor)

    total_loss = style_loss #+ tv_loss #+ gradient_l + ssim_l + psnr_l 

    return total_loss, style_loss, tv_loss, psnr_l, ssim_l #, edge_style_loss,  gradient_l , content_loss, maxSSIM_loss

def loss_function_IQ_Mean_Of_TopK_Min(m, style_layers_lists, input_epigrafh_layers, intervals, weights, input_image_tensor=None, top_k=3):
    style_loss = tf.Variable(0.0, dtype=tf.float32, name="StyleLoss")  # Define style_loss as a variable
    #maxSSIM_loss = tf.Variable(0.0, dtype=tf.float32, name="SSIMLoss")  # Define style_loss as a variable

    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    modified_m = []  # New list to store modified tuples

    filters_widths_shapes = []
    style_layers = style_layers_lists[0]
    for i in range(len(m)):
        texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
        print("texture_filters: ", texture_filters.shape)
        texture_filters_shape = texture_filters.get_shape().as_list()
        width = texture_filters_shape[0]
        filters_widths_shapes.append(width)
        weight = calculate_weight(width, intervals, weights)
        # Create a new tuple with the updated weight
        new_tuple = (m[i][0], weight)
        modified_m.append(new_tuple)

    print('VASILIS filters WIDTH shapes:', filters_widths_shapes)
    print('VASILIS weights:', filters_widths_shapes)

    normalised_m = normalize_weights(modified_m)

    # Convert modified_m back to tuple-like list of tuples
    m_new = tuple(normalised_m)
    print("Modified m as tuples:", m_new)


    for i in range(len(modified_m)):
        style_loss = style_loss + style_loss_per_layer_IQ_TopKMean(style_layers_lists, input_epigrafh_layers, m_new, i, top_k)
        #maxSSIM_loss = maxSSIM_loss + style_loss_per_layer_IQ_MaxSSIM(style_layers_lists, input_epigrafh_layers, m, i)

    #maxSSIM_loss /= len(modified_m)

    tv_loss = tf.Variable(0.0, dtype=tf.float32, name="TVLoss")  # Define style_loss as a variable
    if (input_image_tensor != None):
        tv_loss = tf.image.total_variation(input_image_tensor)

    return style_loss + tv_loss, style_loss, tv_loss#, maxSSIM_loss

def loss_function_IQ_Adaptive_TopK_Min(m, style_layers_lists, input_epigrafh_layers, intervals, weights, input_image_tensor=None, top_k=3, var_threshold=1e-5):
    style_loss = tf.Variable(0.0, dtype=tf.float32, name="Loss")  # Initialize style loss
    
    modified_m = []
    filters_widths_shapes = []

    style_layers = style_layers_lists[0]  # Any reference, just to extract shapes

    # Calculate filter widths and dynamic weights
    for i in range(len(m)):
        texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
        texture_filters_shape = texture_filters.get_shape().as_list()
        width = texture_filters_shape[0]
        filters_widths_shapes.append(width)
        weight = calculate_weight(width, intervals, weights)
        modified_m.append((m[i][0], weight))

    print('VASILIS filters WIDTH shapes:', filters_widths_shapes)
    print('VASILIS weights:', filters_widths_shapes)

    normalised_m = normalize_weights(modified_m)
    m_new = tuple(normalised_m)
    
    top_k_count = 0
    top_min_count = 0

    # Loop over layers and compute adaptive loss
    for i in range(len(m_new)):
        loss_i, top_k_count_i, top_min_count_i = adaptive_style_loss_IQ(style_layers_lists=style_layers_lists, input_epigrafh_layers=input_epigrafh_layers, m=m_new, i=i, k=top_k, var_threshold=var_threshold)
        style_loss = style_loss + loss_i
        top_k_count = top_k_count + top_k_count_i
        top_min_count = top_min_count + top_min_count_i

    tv_loss = tf.image.total_variation(input_image_tensor) if input_image_tensor is not None else 0

    return style_loss + tv_loss, style_loss, tv_loss, top_k_count, top_min_count

def loss_function_IQ_SoftMin(m, style_layers_lists, input_epigrafh_layers, intervals, weights, λ_style_loss = 1e-3, λ_tv = 1e-1, λ_sobel = 1.0, λ_entropy = 0.5, λ_bin = 0.1, input_image_tensor=None, initial_image_tensor=None):
    style_loss = tf.Variable(0.0, dtype=tf.float32, name="Loss")  # Define style_loss as a variable
    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    modified_m = []  # New list to store modified tuples

    filters_widths_shapes = []
    style_layers = style_layers_lists[0]
    for i in range(len(m)):
        texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
        print("texture_filters: ", texture_filters.shape)
        texture_filters_shape = texture_filters.get_shape().as_list()
        width = texture_filters_shape[0]
        filters_widths_shapes.append(width)
        weight = calculate_weight(width, intervals, weights)
        # Create a new tuple with the updated weight
        new_tuple = (m[i][0], weight)
        modified_m.append(new_tuple)

    print('VASILIS filters WIDTH shapes:', filters_widths_shapes)
    print('VASILIS weights:', filters_widths_shapes)

    normalised_m = normalize_weights(modified_m)

    # Convert modified_m back to tuple-like list of tuples
    m_new = tuple(normalised_m)
    print("Modified m as tuples:", m_new)

    for i in range(len(modified_m)):
        style_loss = style_loss + style_loss_per_layer_IQ_SoftMin(style_layers_lists, input_epigrafh_layers, m_new, i)

    #style_loss = tf.log(1 + style_loss)  # compress large values

    tv_loss = tf.image.total_variation(input_image_tensor)

    #sobel_loss = sobel_strength_loss(input_image_tensor)
    #sobel_loss = tf.maximum(sobel_loss, 1e-3) 

    #entr_loss = entropy_loss(input_image_tensor)
    #entr_loss = tf.cast(entr_loss, tf.float32)
    #entr_loss = tf.maximum(entr_loss, 1e-3)

    #bin_loss = binarization_contrast_loss(input_image_tensor)
    #bin_loss = tf.maximum(bin_loss, 1e-3)  # or adjust for your expected scale

    total_loss = (λ_style_loss * style_loss) + (λ_tv * tv_loss) #- (λ_sobel * sobel_loss) - (λ_entropy * entr_loss) - (λ_bin * bin_loss)

    return total_loss, style_loss, tv_loss#, sobel_loss, entr_loss, bin_loss

def loss_function_IQ_Min_Gram(m, style_layers_lists, input_epigrafh_layers):
    style_loss = tf.Variable(0.0, dtype=tf.float32, name="Loss")  # Define style_loss as a variable
    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    for i in range(len(m)):
        style_loss = style_loss + style_loss_per_layer_IQ_Min_Gram(style_layers_lists, input_epigrafh_layers, m, i)

    return style_loss

def loss_function_IQ_Max_Similarity(m, style_layers_lists, input_epigrafh_layers):
    ssim_loss = tf.Variable(0.0, dtype=tf.float32, name="Loss")  # Define style_loss as a variable
    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    for i in range(len(m)):
        ssim_loss = ssim_loss + style_loss_per_layer_IQ_Max_Similarity(style_layers_lists, input_epigrafh_layers, m, i)

    return ssim_loss

def loss_function_IQ_Avg(m, style_layers_lists, input_epigrafh_layers):
    style_loss = tf.Variable(0.0, dtype=tf.float32, name="Loss")  # Define style_loss as a variable
    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    for i in range(len(m)):
        style_loss = style_loss + style_loss_per_layer_IQ_Avg(style_layers_lists, input_epigrafh_layers, m, i)

    return style_loss

def loss_function_IQ_MaxSSIM(m, style_layers_lists, input_epigrafh_layers, intervals, weights, λ_style_loss = 1e-3, λ_tv = 1e-1, input_image_tensor=None,):
    style_loss = tf.Variable(0.0, dtype=tf.float32, name="Loss")  # Define style_loss as a variable
    print("style_layers_lists ", len(style_layers_lists))
    print("style_layers_lists[0] ", len(style_layers_lists[0]))

    modified_m = []  # New list to store modified tuples

    filters_widths_shapes = []
    style_layers = style_layers_lists[0]
    for i in range(len(m)):
        texture_filters = tf.squeeze(style_layers[m[i][0]], 0)
        print("texture_filters: ", texture_filters.shape)
        texture_filters_shape = texture_filters.get_shape().as_list()
        width = texture_filters_shape[0]
        filters_widths_shapes.append(width)
        weight = calculate_weight(width, intervals, weights)
        # Create a new tuple with the updated weight
        new_tuple = (m[i][0], weight)
        modified_m.append(new_tuple)

    print('VASILIS filters WIDTH shapes:', filters_widths_shapes)
    print('VASILIS weights:', filters_widths_shapes)

    normalised_m = normalize_weights(modified_m)

    # Convert modified_m back to tuple-like list of tuples
    m_new = tuple(normalised_m)
    print("Modified m as tuples:", m_new)

    for i in range(len(m)):
        style_loss = style_loss + style_loss_per_layer_IQ_MaxSSIM(style_layers_lists, input_epigrafh_layers, m_new, i)

    tv_loss = tf.image.total_variation(input_image_tensor)
    total_loss = (λ_style_loss * style_loss) + (λ_tv * tv_loss) #- (λ_sobel * sobel_loss) - (λ_entropy * entr_loss) - (λ_bin * bin_loss)

    return total_loss, style_loss, tv_loss#, sobel_loss, entr_loss, bin_loss
