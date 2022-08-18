from cProfile import run
from copyreg import pickle
import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
import tqdm
import glob
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, RocCurveDisplay, auc, log_loss, balanced_accuracy_score
from typing import List, Tuple, Union
import pickle
from tensorflow_addons.optimizers import AdamW

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)


def load_wd_data_paths(fake_main_paths, real_main_paths, no_fake_samples, no_real_samples, seed=0):
    random.seed(seed)
    X_fake_paths = []
    y_fake = []
    X_real_paths = []
    y_real = []
    
    for fake_main_path in fake_main_paths:
        fake_sub_paths = glob.glob(fake_main_path + "*\\")
        for fake_sub_path in fake_sub_paths:
            temp_fakes = []
            for n in range(no_fake_samples):
                temp_fakes.append(fake_sub_path + random.choice(os.listdir(fake_sub_path)))
                
            X_fake_paths.append(temp_fakes)
            y_fake.append(1)
    
    for real_main_path in real_main_paths:
        real_sub_paths = glob.glob(real_main_path + "*\\")
        for real_sub_path in real_sub_paths:
            temp_real = []
            for n in range(no_real_samples):
                temp_real.append(real_sub_path + random.choice(os.listdir(real_sub_path)))
                
            X_real_paths.append(temp_real)
            y_real.append(0)
    
    print(f"WildDeepfake test set contains {len(y_fake)} fake images")
    print(f"WildDeepfake test set contains {len(y_real)} real images")
    
    X_paths = X_fake_paths + X_real_paths
    y = y_fake + y_real

    X_paths, y = shuffle_dataset(X_paths, y, seed)

    return X_paths, y


def read_img(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)


def shuffle_dataset(X, y, seed):
    shuffle_lists = list(zip(X, y))

    random.seed((seed + 1)*2)
    random.shuffle(shuffle_lists)
    X, y = zip(*shuffle_lists)
    
    return list(X), list(y)


def prepare_data(X_paths: List[List[str]], y: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    final_X = []
    for path in tqdm.tqdm(X_paths):
        img_array = []
        for img in path:
            img_array.append(read_img(img))
        final_X.append(img_array)

    final_X = np.array(final_X)
    final_y = np.array(y)

    return final_X, final_y
    

def load_model(model_path: str, checkpoint_path: str = None) -> tf.keras.Model:
    if checkpoint_path != None:
        model = tf.keras.models.load_model(model_path)
        latest_weights = tf.train.latest_checkpoint(checkpoint_path)
        model.load_weights(latest_weights)

    else: 
        model = tf.keras.models.load_model(model_path, custom_objects={'Optimizer': AdamW})

    return model


def batch_predict(X: np.ndarray, model: tf.keras.Model) -> List[List[float]]:
    predictions = list()
    for batch in tqdm.tqdm(X):
        batch_predictions = model.predict(batch, verbose=0)
        predictions.append(batch_predictions)

    return predictions


def infer_from_predictions(predictions: List[List[float]], inference_type: str = "default") -> List[int]:
    inferred_predictions = []  

    if inference_type != "default" and inference_type != "average":
        print(f"Incorrect inference type {inference_type}")
        print(f"Please use 'default' or 'average'")
        return
    
    if inference_type == "default":
        for support in predictions:
            support = (support > 0.5).astype('int32')
            if (support == 1).sum() >= (support == 0).sum():
                inferred_predictions.append(1)
            if (support == 1).sum() < (support == 0).sum():
                inferred_predictions.append(0)

        return inferred_predictions

    else:
        for support in predictions:
            inferred_prediction = sum(support)/len(support)
            inferred_predictions.append((inferred_prediction > 0.5).astype('int32'))
        
        return inferred_predictions


def generate_statistics(y_true: List[int], y_pred: List[int], save_path: str, name: str):
    acc = accuracy_score(y_true, y_pred, normalize=True)
    prec = precision_score(y_true, y_pred)

    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)

    statistics = {
        "name": name,
        "acc": acc,
        "prec": prec,
        "tn": TN,
        "fn": FN,
        "tp": TP,
        "fp": FP,
        "sens": sensitivity,
        "spec": specificity
    }

    statistics = pd.DataFrame(statistics, index=[1])

    os.makedirs(save_path, exist_ok=True)
    cm_analysis(y_true, y_pred, f'{save_path}{name}_cm.png', labels = [1, 0])

    return statistics


def run_original_experiment(data_paths: List[str], model_paths: List[str], number_of_samples: int = 5, seeds: List[int] = 1, inf_type: str = "default"):
    
    all_accs = list()
    all_prec = list()
    all_sens = list()
    all_spec = list()

    main_save_path = "../results/random/" 
    # model_name = model_paths[0].split("\\")[-1].split("_")[1]
    # if ".h5" in model_name:
    #     model_name = model_name.replace(".h5", "")
    if "mlp-mixer" in model_paths[0]:
        model_name = model_paths[0].split("\\")[-2]
    else:
        model_name = model_paths[0].split("\\")[-3]

        if ".h5" in model_name:
            model_name = model_name.replace(".h5", "")


    for sd in seeds:
        print(f"Running experiment for seed {sd}")
        print("Loading data paths...")
        X_paths, y = load_wd_data_paths(data_paths[0], data_paths[1], number_of_samples, number_of_samples, sd*2)


        print(f"Loading image data from paths...")
        X, y = prepare_data(X_paths, y)


        print(f"Loading model...")
        model = load_model(model_paths[0], model_paths[1])
        
        
        print("Predicting classes...")
        raw_predictions = batch_predict(X, model)
        final_predictions = infer_from_predictions(raw_predictions, inf_type)
        del model

        predictions = {
            "raw": raw_predictions,
            "processed": final_predictions
        }

        data_type = f"random_seed_{sd}"
        name = "_".join([model_name, data_type, inf_type, str(number_of_samples), "samples"])
        save_path = f"{main_save_path}{name}/"

        print("Generating statistics...")
        statistics = generate_statistics(y, final_predictions, save_path, name)
        
        all_accs.append(statistics['acc'].values[0])
        all_prec.append(statistics['prec'].values[0])
        all_sens.append(statistics['sens'].values[0])
        all_spec.append(statistics['spec'].values[0])

        print(f"Test: {statistics['name'].values[0]}")
        print(f"Accuracy: {statistics['acc'].values[0]}")
        print(f"Precision: {statistics['prec'].values[0]}")
        print(f"Sensitivity: {statistics['sens'].values[0]}")
        print(f"Specificity: {statistics['spec'].values[0]}")
        print("\n\n")


        print("\nSaving results and statistics\n")
        with open(f"{save_path}{name}_predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)

        with open(f"{save_path}{name}_statistics.pkl", "wb") as f:
            pickle.dump(statistics, f)
    

    general_statistics = {
        "acc": (np.mean(all_accs), np.std(all_accs)),
        "prec": (np.mean(all_prec), np.std(all_prec)),
        "sens": (np.mean(all_sens), np.std(all_sens)),
        "spec": (np.mean(all_spec), np.std(all_spec))
    }

    print(f"Final statistics for experiment with seeds {len(seeds)}")
    print(f"Average accuracy: {general_statistics['acc'][0]} +/- {general_statistics['acc'][1]}")
    print(f"Average precision: {general_statistics['prec'][0]} +/- {general_statistics['prec'][1]}")
    print(f"Average sensitivity: {general_statistics['sens'][0]} +/- {general_statistics['sens'][1]}")
    print(f"Average specificity: {general_statistics['spec'][0]} +/- {general_statistics['spec'][1]}")


    final_save_path = main_save_path + f"general_{model_name}_{inf_type}_stats_seeds_{len(seeds)}_{number_of_samples}_samples/"
    os.makedirs(final_save_path, exist_ok=True)

    with open(final_save_path + "general_statistic.pkl", "wb") as f:
        pickle.dump(general_statistics, f)

    print("Experiment has been finished")
    return general_statistics


def run_experiment(data_paths: List[str], model_paths: List[str], seed: int = 1, inf_type: str = "default"):
    print("Loading data paths...")
    with open(data_paths[0], "rb") as f:
        X_fake = pickle.load(f)

    with open(data_paths[1], "rb") as f:
        X_real = pickle.load(f)  

    N = len(X_fake[0])

    y_fake = [1 for _ in range(len(X_fake))]
    y_real = [0 for _ in range(len(X_real))]

    X_paths = X_fake + X_real
    y = y_fake + y_real


    print(f"Shuffling dataset (seed {1})...")
    X_paths, y = shuffle_dataset(X_paths, y, seed=seed)


    print(f"Loading image data from paths...")
    X, y = prepare_data(X_paths, y)


    print(f"Loading model...")
    model = load_model(model_paths[0], model_paths[1])
    

    print("Predicting classes...")
    raw_predictions = batch_predict(X, model)
    final_predictions = infer_from_predictions(raw_predictions, inf_type)
    del model

    if "mlp-mixer" in model_paths[0]:
        model_name = model_paths[0].split("\\")[-2]
    else:
        model_name = model_paths[0].split("\\")[-3]

        if ".h5" in model_name:
            model_name = model_name.replace(".h5", "")

    data_type = data_paths[0].split("\\")[-1].split("_")[2] + "-dataset"
    name = "_".join([model_name, data_type, inf_type, str(N), "samples"])
    save_path = f"../results/deterministic/{name}/"

    predictions = {
        "raw": raw_predictions,
        "processed": final_predictions
    }

    print("Generating statistics...")
    statistics = generate_statistics(y, final_predictions, save_path, name)
    
    print(f"Test: {statistics['name'].values[0]}")
    print(f"Accuracy: {statistics['acc'].values[0]}")
    print(f"Precision: {statistics['prec'].values[0]}")
    print(f"Sensitivity: {statistics['sens'].values[0]}")
    print(f"Specificity: {statistics['spec'].values[0]}")


    print("\nSaving results and statistics")
    with open(f"{save_path}{name}_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    with open(f"{save_path}{name}_statistics.pkl", "wb") as f:
        pickle.dump(statistics, f)
    

    print("Experiment has been finished")

if __name__ == "__main__":
    wilddeep_fake_path = glob.glob("../../../../DeepFake_Detection/wilddeep_results/fake_test/*/fake/")
    wilddeep_real_path = glob.glob("../../../../DeepFake_Detection/wilddeep_results/real_test/*/real/")
    wilddeep_paths = [wilddeep_fake_path, wilddeep_real_path]
    FULL_EXPERIMENT = True


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)


    if FULL_EXPERIMENT:
        iteration = 0
        models = glob.glob("../analysis/history/models/")
        test_directories = glob.glob("../analysis/datasets/test/paths_only/*/*/")
    
        test_path_pairs = list()
        for directory in test_directories:
            paths = [directory + file for file in os.listdir(directory)]
            test_path_pairs.append(paths)

        for path_pair in test_path_pairs:
            for model in models:
                if iteration != 0:
                    tf.keras.backend.clear_session()
                    gc.collect()
                if not "mlp-mixer" in model:
                    model_name = model.split("\\")[-1].split("_")[1]
                    model_checkpoint = glob.glob(model + f"\\{model_name}_*_checkpoints\\")[0]
                    
                    model_dir = os.listdir(model + "\\saved_models")
                    model_path = model + "\\saved_models\\" + model_dir[0]

                    model_pair = [model_path, model_checkpoint]
                
                if "mlp-mixer" in model:
                    model_path = model + "\\"
                    model_pair = [model_path, None]
                

                run_experiment(path_pair, model_pair, inf_type='average')
                iteration += 1


        # iteration = 0 
        # seeds = [i for i in range(1, 21)]
        # samples = [3, 5, 10]
        # for num_of_samples in samples:
        #     for model in models:
        #         if iteration != 0:
        #             tf.keras.backend.clear_session()
        #             gc.collect()
        #         if not "mlp-mixer" in model:
        #             model_name = model.split("\\")[-1].split("_")[1]
        #             model_checkpoint = glob.glob(model + f"\\{model_name}_*_checkpoints\\")[0]
                    
        #             model_dir = os.listdir(model + "\\saved_models")
        #             model_path = model + "\\saved_models\\" + model_dir[0]

        #             model_pair = [model_path, model_checkpoint]
        #             continue


        #         if "mlp-mixer" in model:
        #             model_path = model + "\\"
        #             model_pair = [model_path, None]
                    

        #         run_original_experiment(wilddeep_paths, model_pair, number_of_samples=num_of_samples, seeds = seeds)

    # fake_path = "../analysis/datasets/old/test_fake_lowest_paths_only_dataset.pkl"
    # real_path = "../analysis/datasets/old/test_real_lowest_paths_only_dataset.pkl"

    # wilddeep_fake_path = glob.glob("../../../DeepFake_Detection/WILDDEEP_DATA/fake_test/*/fake/")
    # wilddeep_real_path = glob.glob("../../../DeepFake_Detection/WILDDEEP_DATA/real_test/*/real/")

    # model_path = "/analysis/history/models/mixed_xception_model/saved_models/xception_clf1024.h5"
    # checkpoint_path = "/analysis/history/models/mixed_xception_model/xception_clf1024_checkpoints"

    # paths = [fake_path, real_path]
    # model_paths = [model_path, checkpoint_path]
    # original_model_paths = [wilddeep_fake_path, wilddeep_real_path]

    # seeds = [i for i in range(1, 11)]

    # run_experiment(paths, model_paths)
    #run_original_experiment(original_model_paths, model_paths, seeds = seeds)