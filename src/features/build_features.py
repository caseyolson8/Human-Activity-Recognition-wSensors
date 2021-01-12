from data.compile_dataset import Activity_Split, Activity, Window
from features.featurizations import *
# from copy_compile_dataset import Activity_Split, Activity, Window
# from featurizations import *
import pickle
import numpy as np
import pandas as pd

ACCEL_AGGS = [get_std, get_RMS, get_ZCR, get_ABSDIFF, get_FFT5, get_spectral]
VIDEO_AGGS_CENTRE = [get_std, get_range, get_ABSDIFF]
VIDEO_AGGS_BOUNDS = [get_height_mean, get_height_std, get_height_range, get_volume_aggs]
N_NANS = len(VIDEO_AGGS_CENTRE)*4 + len(VIDEO_AGGS_BOUNDS)*3


class Featurize(object):
    '''
        wussup Class to create feature matrix in preparation for modelling from list of Window objects with raw time series
    '''
    def __init__(self, window_lst):
        '''
            sets attributes
        '''
        self.raw_windows = window_lst
        self.create_features()
        # self.create_vidfeatures()

    def create_features(self):
        '''
            runs through list of aggregate functions (see globals above) and assembles feature matrix
        '''
        self.activity_labels = []
        self.activity_cats = []
        self.col_labels_accel = []
        feature_matrix_accel = []
        
        self.col_labels_video = []
        feature_matrix_video = []

        first_iter_accel = True
        first_iter_video = True
        for win in self.raw_windows:
            feature_row_accel = []
            feature_row_video = []

            for acc_agg in ACCEL_AGGS:
                if first_iter_accel:
                    data_accel, label_accel = acc_agg(win.accel)
                    self.col_labels_accel.extend(label_accel)
                    feature_row_accel.extend(data_accel)
                else:
                    data_accel, _ = acc_agg(win.accel)
                    feature_row_accel.extend(data_accel)

            if win.has_video:
                if win.video.shape[0] >= 8:
                    centre_data = np.array(win.video.values[:, :3], dtype='float')
                    bounds_data = np.array(win.video.values[:, 3:9], dtype='float')
                    for vid_centre_agg in VIDEO_AGGS_CENTRE:
                        if first_iter_video:
                            data_video, label_video = vid_centre_agg(centre_data)
                            self.col_labels_video.extend(label_video)
                            feature_row_video.extend(data_video)
                        else:
                            data_video, _ = vid_centre_agg(centre_data)
                            feature_row_video.extend(data_video)

                    for vid_bounds_agg in VIDEO_AGGS_BOUNDS:
                        if first_iter_video:
                            data_video, label_video = vid_bounds_agg(bounds_data)
                            self.col_labels_video.extend(label_video)
                            feature_row_video.extend(data_video)
                        else:
                            data_video, _ = vid_bounds_agg(bounds_data)
                            feature_row_video.extend(data_video)
                    first_iter_video = False
                else:
                    feature_row_video.extend([np.nan] * N_NANS)
            else:
                feature_row_video.extend([np.nan] * N_NANS)
            first_iter_accel = False

            feature_matrix_video.append(feature_row_video)
            feature_matrix_accel.append(feature_row_accel)
            self.activity_labels.append(win.name)
            self.activity_cats.append(win.category)
        self.X_accel = np.array(feature_matrix_accel)
        self.X_video = np.array(feature_matrix_video)



if __name__ == "__main__":
    filehandle = open('features/all_data_windowed.obj', 'rb')
    # filehandle = open('src/features/all_data_windowed.obj', 'rb')
    AARP_data = pickle.load(filehandle)

    data_featured = Featurize(AARP_data.windows)
    data_featured.create_features()

    # cleaned_df.to_csv(data_path + 'featurized_data_12022020_3.csv')

    print('complete')

