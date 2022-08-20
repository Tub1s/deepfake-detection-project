import os
import numpy as np
import math
from scipy.stats import wasserstein_distance
from PIL import Image
import pickle
from pathlib import Path
import copy
import cv2 as cv
from typing import List, Union
from collections import deque
import pandas as pd

SMALL_CONST = 1e-10
FEATURE_GENERATORS = [wasserstein_distance]
FEATURE_GENERATORS = {generator.__name__ for generator in FEATURE_GENERATORS}

#TODO: Add exception handling
def generate_histogram(path: str) -> np.ndarray:
    """
    Generates array of histograms of pixel frequency for a given input image.

    Args:
        path (str): Path to the image.

    Returns:
        np.ndarray: Three dimensional array containing pixel value frequencies
        for separate channels in RGB format.
    """    
    image = cv.imread(path)

    bins = 256 # Equal to the number of possible pixel values
    histRange = (0, 256)
    accumulate = False
    
    n_channels = image.shape[2]

    if not (n_channels == 1) or (n_channels == 3):
        raise Exception("Incorrect number of channels in image. \
                         Single or triple channel image required.")


    # Grayscale image case
    if not n_channels == 3:
        hist = np.ravel(cv.calcHist(image, [0], None, [bins], histRange, accumulate=accumulate))
        return hist/hist.sum()


    # Color image case
    bgr_planes = cv.split(image)
    
    # Calculating histograms for separate channels
    b_hist = np.ravel(cv.calcHist(bgr_planes, [0], None, [bins], histRange, accumulate=accumulate))
    g_hist = np.ravel(cv.calcHist(bgr_planes, [1], None, [bins], histRange, accumulate=accumulate))
    r_hist = np.ravel(cv.calcHist(bgr_planes, [2], None, [bins], histRange, accumulate=accumulate))
    
    # Converting counts to frequencies
    # Adding small const to avoid problem with log() calculation for zeros
    b_hist = (b_hist/b_hist.sum()) + SMALL_CONST
    g_hist = (g_hist/g_hist.sum()) + SMALL_CONST
    r_hist = (r_hist/r_hist.sum()) + SMALL_CONST
        
    return np.array([r_hist, g_hist, b_hist])


#Calculate average Wasserstein distance for sequence of images
#TODO: Add custom dynamics metrics
#TODO: Modify to calculate k histograms only once, and then store values?
#TODO: Return pd.DataFrame with cols: [vid_num, seq_num, first_frame, seq_len, img_type, feature1, ... featuren]
def generate_features(list_of_image_paths: List[str], seq_len: int, 
                      feature_type: Union[str, List[str]]) -> np.ndarray:
    if seq_len < 2:
        raise Exception("Sequence length has to be greater or equal to 2")

    if not set(feature_type).issubset(set(FEATURE_GENERATORS.keys())):
        raise Exception(f"Unknown features {list(set(feature_type) - set(FEATURE_GENERATORS.keys()))}.")

    histograms_cache = dict()
    distances = list()
    img_indices = deque()

    sequence = list_of_image_paths[0].split("/")[-2]
    sample_type = list_of_image_paths[0].split("/")[-3]
    video = list_of_image_paths[0].split("/")[-4]
    dataset_subset = list_of_image_paths[0].split("/")[-5].split("_")[1]

    for i in range(len(list_of_image_paths) - seq_len + 1):
        if not histograms_cache.keys():
            for k in range(seq_len):
                img_indices.append(int(list_of_image_paths[i+k].split("/")[-1].replace(".png", "")))
                histograms_cache[k] = generate_histogram(list_of_image_paths[i+k])

        if i+seq_len-1 not in histograms_cache.keys():
            img_indices.append(int(list_of_image_paths[i+seq_len-1].split("/")[-1].replace(".png", "")))
            histograms_cache[i+seq_len-1] = generate_histogram(list_of_image_paths[i+seq_len-1])
        
        first_frame = img_indices.popleft()   
        sample_data = {"subset": dataset_subset,
                       "type": sample_type,
                       "video": video,
                       "sequence": sequence,
                       "first_frame": first_frame,
                       "sequence_length": seq_len
        }

        for feature in feature_type:
            sequence_average_feature = 0.0
            for key in histograms_cache.keys():
                next_key = key + 1
                if next_key not in histograms_cache.keys():
                    break
                
                average_feature = 0.0
                n_channels = histograms_cache[key].shape[0]

                for j in range(n_channels):
                    average_feature += (FEATURE_GENERATORS[feature](histograms_cache[key][j], histograms_cache[next_key][j]))/n_channels
                
                sequence_average_feature += average_feature/(seq_len-1)
        
            sample_data[feature] = sequence_average_feature
        
        distances.append(pd.Series(sample_data))
        del histograms_cache[i]  

    return np.array(distances)

#Find original frames given result index
def find_org_frames(list_of_image_paths: List[str], result_index: int, sequence_length: int):
    frame_paths = list()

    for i in range(result_index, result_index + sequence_length):
        frame_paths.append(list_of_image_paths[i])


    return frame_paths


# Save result for given seq as: seq_data-seq_len.npy
def save_avg_distance_seq(main_path, seq_len):
    results_main_path = "../../../../DeepFake_Detection/wilddeep_results/"
    results_path = copy.copy(results_main_path)

    split_seq_path = Path(main_path).parts[::-1]
    split_seq_path = split_seq_path[0:4]
    split_seq_path = split_seq_path[::-1]

    for part in split_seq_path:
        results_path = results_path + part + "/"
    
    os.makedirs(results_path, exist_ok = True)
    

    list_of_image_paths = os.listdir(main_path)
    list_of_image_paths = sorted([int(x.replace('.png', '')) for x in list_of_image_paths])
    list_of_image_paths = [main_path.replace("\\", "/") + str(img) + ".png" for img in list_of_image_paths]
    
    distances = generate_features(list_of_image_paths, seq_len)

    idx_paths_pairs = {idx: find_org_frames(list_of_image_paths, idx, seq_len) 
                       for idx in range(distances.shape[0])}

    
    with open(f"{results_path}/seq_data-{seq_len}.npy", 'wb') as f:
        np.save(f, distances)
    
    with open(f"{results_path}/idx_frames-{seq_len}.npy", 'wb') as f:
        pickle.dump(idx_paths_pairs, f)


# calculate the kl divergence (is not symmetrical) requires non-zero inputs in distributions
def kl_divergence(p, q):
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence (is symmetrical)
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

