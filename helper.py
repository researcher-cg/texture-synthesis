from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.cluster
from skimage import exposure
from skimage.transform import match_histograms
from skimage.transform import pyramid_gaussian
import binascii
import struct
import random

def resize_and_rescale_img(image_path, w, h, output_path_, output_filename):
    # This will resize the image to width x height dimensions and then scale down in the range of [0-1]
    if os.path.isfile(image_path):
        img = Image.open(image_path)
        img_resized = img.resize(size=(w, h))
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        img_resized.save(output_path_ + output_filename)
        print("\nimg_resized: ", img_resized)
        img_array = np.array(img_resized, dtype=np.float32)
        print("\nimg_array: ", img_array.shape,"\n")

        img_array = np.expand_dims(img_array, 0)
        print("\nimg_array: ", img_array.shape,"\n")


        img_array = img_array / 255.0

        return img_array
    else:
        print("No image found in given location.")

def fix_img(image_path):
    # This will resize the image to width x height dimensions and then scale down in the range of [0-1]
    if os.path.isfile(image_path):
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32)
        print("\nimg_array: ", img_array.shape,"\n")

        img_array = np.expand_dims(img_array, 0)
        print("\nimg_array: ", img_array.shape,"\n")

        img_array = img_array / 255.0

        return img_array
    else:
        print("No image found in given location.")

def post_process_and_display(cnn_output, output_path, output_filename, source_img_org, save_file=True):
     # This will take input_noise of (1, w, h, channels) shapped array taken from tensorflow operation
    # and ultimately displays the image

    x = np.squeeze(cnn_output)
    y = np.squeeze(source_img_org)
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))

    x *= 255.0
    x = np.clip(x, 0.0, 255.0).astype('uint8')
    y *=255.0
    y = np.clip(y, 0.0, 255.0).astype('uint8')

    #new_texture = x['x'].reshape(*cnn_output.shape[1:]).transpose(1,2,0)[:,:,::-1]
    #print("x: ", x.shape , " source: ", y.shape)
    #print("x: ", x[0] , "\n source: ", y[0])

    new_texture = histogram_matching(x, y)

    img = Image.fromarray(new_texture, mode='RGB')
    # img.show()
    if save_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img.save(output_path + output_filename)

    return new_texture

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram

    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))

def histogram_matching_scipy(org_image, match_image):
    matched = match_histograms(org_image, match_image, multichannel=True)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(org_image)
    ax1.set_title('Source')
    ax2.imshow(match_image)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show(block=False)
    fig1, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

    for i, img in enumerate((org_image, match_image, matched)):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[..., c])
            axes[c, i].plot(bins, img_cdf)
            axes[c, 0].set_ylabel(c_color)

    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')

    plt.tight_layout()
    # plt.show(Image)

def histogram_matching(org_image, match_image, grey=False, n_bins=255):
        '''
        Matches histogram of each color channel of org_image with histogram of match_image

        :param org_image: image whose distribution should be remapped
        :param match_image: image whose distribution should be matched
        :param grey: True if images are greyscale
        :param n_bins: number of bins used for histogram calculation
        :return: org_image with same histogram as match_image
        '''
        histogram_matching_scipy(org_image, match_image)

        if grey:
            hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image.ravel()))
            r[r>cum_values.max()] = cum_values.max()
            matched_image = inv_cdf(r).reshape(org_image.shape)
        else:
            matched_image = np.zeros_like(org_image)
            for i in range(3):
                hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
                cum_values = np.zeros(bin_edges.shape)
                cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
                inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
                r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
                r[r>cum_values.max()] = cum_values.max()
                matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)

        return matched_image

def most_and_less_frequent_color(image, numOfClusters, iterations, init):
    print('reading image')
    im = Image.open(image)
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    centroids, labels = scipy.cluster.vq.kmeans2(ar, numOfClusters, iter=iterations, minit=init)

    vecs, dist = scipy.cluster.vq.vq(ar, centroids)         # assign codes
    # print('Vectors:\n', vecs, 'vecs: ', vecs.shape)
    counts, bins = scipy.histogram(vecs, len(centroids))    # count occurrences
    # print('Centres:\n', centroids)
    # print('Bins:\n', bins)
    # print('\nCounts: ', counts, '\n')

    index_max = scipy.argmax(counts)                    # find most frequent
    most = centroids[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in most)).decode('ascii')
    print('most frequent is %s -> #%s ' % (most, colour), counts[index_max])

    index_min = scipy.argmin(counts)                    # find less frequent
    less = centroids[index_min]
    colour = binascii.hexlify(bytearray(int(c) for c in less)).decode('ascii')
    print('less frequent is %s -> #%s ' % (less, colour), counts[index_min])

    sum = np.sum(counts)
    # print('sum counts: ', sum)
    return centroids, counts, labels, counts/sum

def roulette(inputs, weights):
    r = random.uniform(0, sum(weights))
    # loop through a list of inputs and max cutoff values, returning
    # the first value for which the random num r is less than the cutoff value
    for n,v in map(None, inputs,[sum(weights[:x+1]) for x in range(len(weights))]):
        if r < v:
            return n

def gaussian_pyramid(texture_array, max_layer=4, downscale=2, multichannel=True):
    image = np.squeeze(texture_array)
    print("\n NEW IMAGE: ",image.shape)
    rows, cols, dim = image.shape
    pyramid = tuple(pyramid_gaussian(image, max_layer, downscale, multichannel))

    composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

    composite_image[:rows, :cols, :] = pyramid[0]

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    fig, ax = plt.subplots()
    ax.imshow(composite_image)
    plt.show(block=False)
    return pyramid