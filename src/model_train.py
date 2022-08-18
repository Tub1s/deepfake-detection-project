import os
import numpy as np
import pandas as pd
import random
import tqdm
import glob
import cv2
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from datetime import datetime
from typing import List, Union, Tuple
import pickle
import gc


LOAD_DFDC = True 
LOAD_WILD = True
LOAD_CELE = True

DFDC_SEED = 1
VERSION = 224

if LOAD_DFDC:
    LABELS = ['REAL','FAKE']
    dfdc_paths = pd.read_json("../../../DeepFake_Detection/DFDC_ALL_DATA/metadata/metadata.json")

if LOAD_WILD:
    wild_fake_train_main_paths = glob.glob("../../../DeepFake_Detection/WILDDEEP_DATA/fake_train/*/fake/")
    wild_real_train_main_paths = glob.glob("../../../DeepFake_Detection/WILDDEEP_DATA/real_train/*/real/")
    
if LOAD_CELE:
    LABELS = ['REAL','FAKE']
    cele_paths = pd.read_json("../../../DeepFake_Detection/CELEB-DF_ALL_DATA/metadata/metadata.json")


def read_img(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)


def join_lists(X_paths_1, y_1, X_paths_2, y_2, seed):
    X_paths = X_paths_1 + X_paths_2
    y = y_1 + y_2
    
    shuffle_lists = list(zip(X_paths, y))
    random.seed((seed + 1)*2)
    random.shuffle(shuffle_lists)
    
    X_paths, y = zip(*shuffle_lists)
    
    return list(X_paths), list(y)


def get_path(x, dataset = 'dfdc', version = 224):
    if version == 224 and dataset == 'dfdc':
        path="../../../DeepFake_Detection/DFDC_ALL_DATA_224/" + x.replace('.mp4', '') + '.jpg'

    elif version == 256 and dataset == 'dfdc':
        path="../../../DeepFake_Detection/DFDC_ALL_DATA/" + x.replace('.mp4', '') + '.jpg'
        
    elif version == 224 and dataset == 'cele':
        path="../../../DeepFake_Detection/CELEB-DF_ALL_DATA_224/" + x
        
    elif version == 256 and dataset == 'cele':
        path="../../../DeepFake_Detection/CELEB-DF_ALL_DATA/" + x
    
    if not os.path.exists(path):
       raise Exception
    return path


def load_data_from_file(fake_file_path: str, real_file_path: str, seed: int = 0) -> Tuple[List[str], List[str]]:
    with open(fake_file_path, "rb") as f:
        fake_data = pickle.load(f)
    
    with open(real_file_path, "rb") as f:
        real_data = pickle.load(f)

    random.seed(seed)
    X_fake_paths = []
    y_fake = []
    X_real_paths = []
    y_real = []

    for fake_seq in fake_data:
        for fake_path in fake_seq['paths']:
            X_fake_paths.append(fake_path)
            y_fake.append(1)
    

    for real_seq in real_data:
        for real_path in real_seq['paths']:
            X_real_paths.append(real_path)
            y_real.append(0)

    X_paths = X_fake_paths + X_real_paths
    y = y_fake + y_real

    shuffle_lists = list(zip(X_paths, y))
    random.seed((seed + 1)*2)
    random.shuffle(shuffle_lists)
    
    X_paths, y = zip(*shuffle_lists)
    
    return list(X_paths), list(y)


def load_wd_data(fake_main_paths, real_main_paths, no_fake_samples, no_real_samples, seed=0):
    print('Loading WildDeepfake Dataset paths')
    random.seed(seed)
    X_fake_paths = []
    y_fake = []
    X_real_paths = []
    y_real = []
    for fake_main_path in fake_main_paths:
        fake_sub_paths = glob.glob(fake_main_path + "*\\")
        for fake_sub_path in fake_sub_paths:
            for n in range(no_fake_samples):
                X_fake_paths.append(fake_sub_path + random.choice(os.listdir(fake_sub_path)))
                y_fake.append(1)
    
    for real_main_path in real_main_paths:
        real_sub_paths = glob.glob(real_main_path + "*\\")
        for real_sub_path in real_sub_paths:
            for n in range(no_real_samples):
                X_real_paths.append(real_sub_path + random.choice(os.listdir(real_sub_path)))
                y_real.append(0)
    
    print(f"Loaded {len(y_fake)} fake samples from WildDeepfake dataset")
    print(f"Loaded {len(y_real)} real samples from WildDeepfake dataset")
    print(f'Current WildDeepfake fake/real ratio is equal to {len(y_fake)/(len(y_fake)+len(y_real))*100}% / {len(y_real)/(len(y_fake)+len(y_real))*100}%')
    X_paths = X_fake_paths + X_real_paths
    y = y_fake + y_real

    shuffle_lists = list(zip(X_paths, y))
    random.seed((seed + 1)*2)
    random.shuffle(shuffle_lists)
    
    X_paths, y = zip(*shuffle_lists)
    print('WildDeepfake Dataset paths loading has been finished')
    
    return list(X_paths), list(y)


def load_dfdc_data():
    print('Loading DFDC Dataset paths')
    sleep(0.1)
    X_paths, y = [], []
    images = list(dfdc_paths.columns.values)
    for x in tqdm.tqdm(images):
        try:
            X_paths.append(get_path(x, dataset='dfdc', version = VERSION))
            y.append(LABELS.index(dfdc_paths[x]['label']))
        except Exception as err:
            #print(err)
            pass
        
    print(f"Loaded {y.count(1)} fake samples from DFDC dataset")
    print(f"Loaded {y.count(0)} real samples from DFDC dataset")
    print(f'Current DFDC fake/real ratio is equal to {y.count(1)/(y.count(1)+y.count(0))*100}% / {y.count(0)/(y.count(1)+y.count(0))*100}%')
    shuffle_dfdc = list(zip(X_paths, y))
    random.seed(DFDC_SEED*4)
    random.shuffle(shuffle_dfdc)
    X_paths, y = zip(*shuffle_dfdc)
    print('DFDC Dataset paths loading has been finished')
    
    return list(X_paths), list(y)


def resample_data(X, y, undersample = False, oversample = False, under_sampling_strategy = 1, over_sampling_strategy = 1, seed = 5):
    if undersample or oversample:
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        if undersample:
            print('Undersampling in progress...')
            under = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state = 5945 * seed)
            X, y = under.fit_resample(X, y)
            print('Undersampling has been finished')
        if oversample:
            print('Oversampling in progress...')
            over = RandomOverSampler(sampling_strategy=under_sampling_strategy, random_state = 581 * seed)
            X, y = over.fit_resample(X, y)
            print('Oversampling has been finished')
    
        X = X.reshape(1, -1)
        y = y.reshape(1, -1)
        X = X.tolist()
        y = y.tolist()
        X = X[0]
        y = y[0]
    
        shuffle_lists = list(zip(X, y))
        random.seed((seed + 1)*58)
        random.shuffle(shuffle_lists)
    
        X, y = zip(*shuffle_lists)
        X = list(X)
        y = list(y)
    
    return X, y

# OVERSAMPLE_MINOR = False
# UNDERSAMPLE_MAJOR = True
# OVERSAMLPING_STRATEGY = 1
# UNDERSAMPLING_STRATEGY = 1
# RESAMPLE_SEED = 1
# MERGING_SEED = 1

# fake_lowest_path = "../analysis/datasets/train/lowest/paths_only/train_fake_lowest_paths_only_dataset.pkl"
# fake_highest_path = "../analysis/datasets/train/highest/paths_only/train_fake_highest_paths_only_dataset.pkl"
# real_lowest_path = "../analysis/datasets/train/lowest/paths_only/train_real_lowest_paths_only_dataset.pkl"
# real_highest_path = "../analysis/datasets/train/highest/paths_only/train_real_highest_paths_only_dataset.pkl"

# X_dfdc_paths, y_dfdc = load_dfdc_data()
# X_dfdc_paths, y_dfdc = resample_data(X_dfdc_paths, y_dfdc, undersample = UNDERSAMPLE_MAJOR, oversample = OVERSAMPLE_MINOR, under_sampling_strategy = UNDERSAMPLING_STRATEGY, over_sampling_strategy = OVERSAMLPING_STRATEGY, seed = 5*RESAMPLE_SEED)
# print(f"Rebalanced DFDC dataset contains {y_dfdc.count(1)} fake samples and {y_dfdc.count(0)} real samples")


# X_wild_paths, y_wild = load_data_from_file(fake_lowest_path, real_lowest_path, seed = 1)
# print('\nMerging and shuffling DFDC Dataset paths and WildDeepfake Dataset paths...')
# X_paths, y = join_lists(X_dfdc_paths, y_dfdc,
#                         X_wild_paths, y_wild,
#                         seed = 2*MERGING_SEED)
# print('Process has been finished')
# print(f'Final DFDC+WildDeepfake fake/real ratio is equal to {y.count(1)/(y.count(1)+y.count(0))*100}% / {y.count(0)/(y.count(1)+y.count(0))*100}%')

