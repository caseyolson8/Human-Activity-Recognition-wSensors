import numpy as np
import pandas as pd 
import pickle
from scipy.fft import fft, ifft


def get_mean(window):
    '''
        Average values
    '''
    labels = ['Mean_A', 'Mean_X', 'Mean_Y', 'Mean_Z']
    (x_mean, y_mean, z_mean) = window.mean(axis=0)
    a_mean = np.linalg.norm(window, axis=1).mean()
    return [a_mean, x_mean, y_mean, z_mean], labels


def get_std(window):
    '''
        Standard Deviation
    '''
    labels = ['Std_A', 'Std_X', 'Std_Y', 'Std_Z']
    (x_std, y_std, z_std) = np.sqrt(window.var(axis=0))
    a_std = np.sqrt(np.linalg.norm(window, axis=1).var())
    return [a_std, x_std, y_std, z_std], labels


def get_range(window):
    '''
        Returns the span of the data
    '''
    labels = ['Width_A', 'Width_X', 'Width_Y', 'Width_Z']
    (x_range, y_range, z_range) = np.ptp(window, axis=0)
    a_range = np.ptp(np.linalg.norm(window, axis=1))
    return [a_range, x_range, y_range, z_range], labels


def get_RMS(window):
    '''
        root mean square
    '''
    labels = ['RMS_A', 'RMS_X', 'RMS_Y', 'RMS_Z']
    x_rms, y_rms, z_rms = np.sqrt((window**2).sum(axis=0)/window.shape[0])
    a_rms = np.sqrt((np.linalg.norm(window, axis=1)**2).sum()/window.shape[0])
    return [a_rms, x_rms, y_rms, z_rms], labels


def get_ZCR(window):
    '''
        Zero-crossing rate
    '''
    labels = ['ZCR_A', 'ZCR_X', 'ZCR_Y', 'ZCR_Z']
    window2 = np.concatenate((np.linalg.norm(window, axis=1).reshape(-1,1), window), axis=1)
    win_mean_norm = (window2 - window2.mean(axis=0))
    lag = np.sign(win_mean_norm[0,:])
    counts = np.array([0,0,0,0])
    for row in win_mean_norm[1:,:]:
        counts += (np.sign(row)!=lag).astype(int)
        lag = np.sign(row)
    return counts, labels


def get_ABSDIFF(window):
    '''
        absolute difference from the mean value
    '''
    labels = ['ABS_A', 'ABS_X', 'ABS_Y', 'ABS_Z']
    window2 = np.concatenate((np.linalg.norm(window, axis=1).reshape(-1,1), window), axis=1)
    win_mean_norm = np.absolute(window2 - window2.mean(axis=0))
    return win_mean_norm.sum(axis=0)/window.shape[0], labels


def get_FFT5(window):
    '''
        First 5 Fourier coefficients
    '''
    label_base = ['FFT5_A', 'FFT5_X', 'FFT5_Y', 'FFT5_Z']
    coeff_n = ['_0', '_1', '_2', '_3', '_4']
    labels = []
    for each in label_base:
        labels.extend((list(np.char.add(each, coeff_n))))
    xyz = []
    for column in window.T:
        xyz.append(np.absolute(fft(column)[:5]))
    a_fft5 = np.absolute(fft(np.linalg.norm(window, axis=1))[:5])
    x_fft5, y_fft5, z_fft5 = tuple(xyz)
    return list(a_fft5) + list(x_fft5) + list(y_fft5) + list(z_fft5), labels


def get_spectral(window):
    '''
        Spectral energy
    '''
    labels = ['Energy_A', 'Energy_X', 'Energy_Y', 'Energy_Z']
    xyz = []
    for column in window.T:
        c_m = np.absolute(fft(column))**2
        n = len(c_m)
        xyz.append(c_m.sum()/n)
    c_m = np.absolute(fft(np.linalg.norm(window, axis=1)))**2
    n = len(c_m)
    a_e = c_m.sum()/n
    x_e, y_e, z_e = tuple(xyz)
    return [a_e, x_e, y_e, z_e], labels


def get_height_mean(video_window):
    '''
        Calculates the height, width, depth and returns the mean over the windowed time-series
    '''
    brb, flt = np.split(video_window, 2, axis=1)
    heights = np.absolute(brb-flt)
    labels = ['bbx_mean', 'bby_mean', 'bbz_mean']
    return list(heights.mean(axis=0)), labels

def get_height_std(video_window):
    '''
        Calculates the height, width, depth and returns the std over the windowed time-series
    '''
    brb, flt = np.split(video_window, 2, axis=1)
    heights = np.absolute(brb-flt)
    labels = ['bbx_std', 'bby_std', 'bbz_std']
    return list(np.sqrt(list(heights.var(axis=0)))), labels


def get_height_range(video_window):
    '''
        Calculates the height, width, depth and returns the range over the windowed time-series
    '''
    brb, flt = np.split(video_window, 2, axis=1)
    heights = np.absolute(brb-flt)
    labels = ['bbx_range', 'bby_range', 'bbz_range']
    return list(np.ptp(heights, axis=0)), labels


def get_volume_aggs(video_window):
    '''
        Calculates the volume and returns several statistical aggregates over the windowed time-series
    '''
    brb, flt = np.split(video_window, 2, axis=1)
    volumes = np.absolute(np.prod(brb-flt, axis=1))
    labels = ['Volume_mean', 'Volume_std', 'Volume_range']

    return [volumes.mean(), volumes.std(), np.ptp(volumes)], labels
    

if __name__ == "__main__":
    pass
    # filehandle = open('second_try.obj', 'rb')
    # data = pickle.load(filehandle)
    # data.filtered[0].accel.iloc[:20].values
    # from src.features.featurizations import *