import os
import numpy as np
import math
from scipy.stats import wasserstein_distance
from PIL import Image
import pickle
from pathlib import Path
import copy

NUM_CHANNELS = 3

#Read image as a numpy array
def read_image(path):
    image = Image.open(path)
    return np.asarray(image)

#Create histogram from an image
#! Add custom number of bins support in future
def get_histograms(image):
    histograms = list()
    for channel in range(image.shape[2]):
        histogram, bin_edges = np.histogram(
            image[:, :, channel], bins=256, range=(0, 256)
        )
        histogram = [i/sum(histogram) for i in histogram]
        histograms.append(histogram)

    return histograms

#Calculate average distance for pairs of frames
def get_avg_distance(main_path, list_of_images):
    cache = dict()
    distances = list()
    for i in range(len(list_of_images) - 1):
        if not cache.keys():
            cache[i] = read_image(f"{main_path}{str(list_of_images[i])}.png")
            cache[i+1] = read_image(f"{main_path}{str(list_of_images[i+1])}.png")

        if str(i+1) not in cache.keys():
            cache[i+1] = read_image(f"{main_path}{str(list_of_images[i+1])}.png")

        histogram_1 = get_histograms(cache[i])
        histogram_2 = get_histograms(cache[i+1])
        
        avg_distance = 0.0
        for j in range(NUM_CHANNELS):
            avg_distance += (wasserstein_distance(histogram_1[j], histogram_2[j]))/NUM_CHANNELS

        del cache[i]
        distances.append(avg_distance)
    
    return distances

#Calculate average Wasserstein distance for sequence of images
def get_avg_distance_seq(main_path, list_of_images, seq_len):
    cache = dict()
    cache_histograms = dict()
    distances = list()

    if seq_len < 2:
        print("Sequence length has to be greater or equal 2")
        return

    for i in range(len(list_of_images) - seq_len + 1):
        if not cache.keys():
            for k in range(seq_len):
                cache[k] = read_image(f"{main_path}{str(list_of_images[i+k])}.png")

        if str(i+seq_len-1) not in cache.keys():
            cache[i+seq_len-1] = read_image(f"{main_path}{str(list_of_images[i+seq_len-1])}.png")
        
        if not cache_histograms.keys():
            for key in cache.keys():
                cache_histograms[key] = get_histograms(cache[key])

        if str(i+seq_len-1) not in cache_histograms.keys():
            cache_histograms[i+seq_len-1] = get_histograms(cache[i+seq_len-1])

        #print(f"iter: {i}; frames: {cache.keys()}")

        seq_avg_distance = 0.0
        
        for key in cache_histograms.keys():
            next_key = key + 1
            if next_key not in cache_histograms.keys():
                break
            
            avg_distance = 0.0

            for j in range(NUM_CHANNELS):
                avg_distance += (wasserstein_distance(cache_histograms[key][j], cache_histograms[next_key][j]))/NUM_CHANNELS
            
            seq_avg_distance += avg_distance/(seq_len-1)
        
        del cache[i]
        del cache_histograms[i]
        distances.append(seq_avg_distance)

    return distances

#Find original frames given result index
def find_org_frames(main_path, list_of_images, 
                    result_index, seq_len):

    frame_paths = list()

    for i in range(result_index, result_index + seq_len):
        frame_paths.append(main_path + str(list_of_images[i]) + '.png')


    return frame_paths


# Save result for given seq as: seq_data-seq_len.npy
def save_avg_distance_seq(seq_len, main_path):
    results_main_path = "../../../../DeepFake_Detection/wilddeep_results/"
    results_path = copy.copy(results_main_path)

    split_seq_path = Path(main_path).parts[::-1]
    split_seq_path = split_seq_path[0:4]
    split_seq_path = split_seq_path[::-1]

    for part in split_seq_path:
        results_path = results_path + part + "\\"
    
    os.makedirs(results_path, exist_ok = True)
    

    list_of_images = os.listdir(main_path)
    list_of_images = sorted([int(x.replace('.png', '')) for x in list_of_images])
    
    distances = get_avg_distance_seq(main_path, list_of_images, seq_len)
    distances = np.array(distances)

    idx_paths_pairs = {idx: find_org_frames(main_path, list_of_images, idx, seq_len) 
                       for idx in range(distances.shape[0])}

    
    with open(f"{results_path}\\seq_data-{seq_len}.npy", 'wb') as f:
        np.save(f, distances)
    
    with open(f"{results_path}\\idx_frames-{seq_len}.npy", 'wb') as f:
        pickle.dump(idx_paths_pairs, f)


# calculate the kl divergence (is not symmetrical)
def kl_divergence(p, q):
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence (is symmetrical)
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

