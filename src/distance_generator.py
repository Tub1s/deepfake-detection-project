import glob
import multiprocessing
import features.sequence_dynamic as seq_dyn
from functools import partial

def main():
    fake_main_paths = glob.glob("../../../DeepFake_Detection/WILDDEEP_DATA/fake_test/*/fake/")
    real_main_paths = glob.glob("../../../DeepFake_Detection/WILDDEEP_DATA/real_test/*/real/")
    save_path = "../../../DeepFake_Detection/wilddeep_results/"
    all_paths = list()
    for fake_main_path in fake_main_paths:
            fake_sub_paths = glob.glob(fake_main_path + "*\\")
            all_paths += fake_sub_paths

    for real_main_path in real_main_paths:
            real_sub_paths = glob.glob(real_main_path + "*\\")
            all_paths += real_sub_paths
    
    pool = multiprocessing.Pool()
    pool.map(partial(seq_dyn.save_avg_distance_seq, 
                     save_path=save_path, 
                     sequence_length=10, 
                     feature_type=["wasserstein_distance", "jensenshannon"]), all_paths)
    pool.close()


if __name__ == "__main__":
    main()