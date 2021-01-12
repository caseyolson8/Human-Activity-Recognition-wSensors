import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import json
import os

plt.style.use('ggplot')

ACTIVITY_PREFIX = {'a_': 'Ambulation',
                   'p_': 'Posture',
                   't_': 'Transition'}


class Data_Sequence(object):
    """
    A Class to read in a single 'recording' of 'training data' in the 'SPHERE Challenge'.

    Methods
    ----------
        load_data(self)
            loads raw data into accessible attributes, where pathway is defined by 'self.path'
    
        load_annotations(self)
            loads the annotations data from a single observer.  Target variables are derived from this
    """

    def __init__(self, meta_root, data_path):
        """
        Constructs all the necessary (and some unused) attributes for the Data_Sequence object.

        Parameters
        ----------
            meta_root (str)
                pathway to metadata with raw data labels in .json
            data_path (str)
                pathway to folder with raw data series

        """

        self.path = data_path
        video_cols = json.load(open(os.path.join(meta_root, 'video_feature_names.json')))  # Video feature names ##
        self.centre_3d = video_cols['centre_3d']
        self.bb_3d = video_cols['bb_3d']
        self.video_names = json.load(open(os.path.join(meta_root, 'video_locations.json')))

        self.meta = json.load(open(os.path.join(data_path, 'meta.json')))           # Other features & metadata ##
        self.acceleration_keys = json.load(open(os.path.join(meta_root, 'accelerometer_axes.json')))
        self.rssi_keys = json.load(open(os.path.join(meta_root, 'access_point_names.json')))
        self.pir_names = json.load(open(os.path.join(meta_root, 'pir_locations.json')))
        self.location_targets = json.load(open(os.path.join(meta_root, 'rooms.json')))
        self.activity_targets = json.load(open(os.path.join(meta_root, 'annotations.json')))

    def load_data(self):
        """
        Loads raw data from folder 'self.path'
        """
        self.acceleration = self.load_accelerations()
        self.videos_lst = self.load_videos()
        self.annotation = self.load_annotations()

    def load_accelerations(self):
        """
        Loads acceleration data, all 3 axes, into Dataframe with time as index
        """
        accel = pd.read_csv(self.path + 'acceleration.csv', index_col='t')
        return accel[self.acceleration_keys]

    def load_videos(self):
        """
        Loads video derived position data into Dataframe with time as index.

        Returns
            Dataframe with time as index and columns with bounding box & center of mass
             coordinates
        """
        columns = self.centre_3d + self.bb_3d
        hallway_video = pd.read_csv(self.path + 'video_hallway.csv', index_col='t')[columns]
        hallway_video['label'] = 'Hallway'
        kitchen_video = pd.read_csv(self.path + 'video_kitchen.csv', index_col='t')[columns]
        kitchen_video['label'] = 'Kitchen'
        livingroom_video = pd.read_csv(self.path + 'video_living_room.csv', index_col='t')[columns]
        livingroom_video['label'] = 'Living_Room'
        return [hallway_video, kitchen_video, livingroom_video]

    def load_annotations(self):
        """
        Loads annotation data from the first observer
        """
        annotations_file_name = self.path + 'annotations_0.csv'
        return pd.read_csv(annotations_file_name)


class Activity_Split(object):
    """
    This class takes a Data_Sequence object & splits and manages it into series of single activities

    Methods
    ----------
        load_data(self)
            loads raw data into accessible attributes, where pathway is defined by 'self.path'

        load_annotations(self)
            loads the annotations data from a single observer.  Target variables are derived from this
    """

    def __init__(self):
        '''
        Constructs all necessary attributes for Acitivity_Split object.  Takes no arguments
        '''
        self.activity_count = defaultdict(int)
        self.activity_lengths = defaultdict(list)
        self.activities = []
        self.filtered = []
        self.filter_on = None

    def add_data(self, data):
        '''
        Splits and appends data from a Data_Sequence object into a list (self.activities) of individual 'Activity' objects 

        Parameters
        ----------
            data (Data_Sequence)
                Data Sequence object from which to pull labelled accelerometer data puts each 
        '''
        for _, row in data.annotation.iterrows():
            self.activities.append(Activity(row))
            self.activities[-1].grab_data(data)
            self.activity_count[row['name']] += 1
            self.activity_lengths[row['name']].append(row['end'] - row['start'])

    def filter_data(self, action_list, t_window=1, t_shift=0.5):
        '''
            Filters 'Activity' objects from lists 'self.activities' into 'self.filtered' 
                if they are in action_list with time span > t_length.

        Parameters
        ----------
            action_list (list)
                list of strings with activities to keep
            t_length (a numeric type)
                time in seconds
        '''
        self.filtered = []
        for act in self.activities:
            if (act.name in action_list) and (act.span > t_window):
                self.filtered.append(act)
        self.filter_on = action_list
        self.windows = []
        self._create_windows_(t_window, t_shift)

    def _create_windows_(self, t_window, t_shift, dt=0.05):
        '''
            Break single activity into multiple windows of length 't_window' and defined by 't_shift'

        Parameters
        -----------
            t_window (numeric)
                width of the window
            dt (float)
                signal sampling rate
        '''
        len_window = int(np.floor(t_window / dt))
        t_shift = int(t_shift / dt)
        for each in self.filtered:
            num_points = each.accel.shape[0]
            num_windows = int(np.floor((num_points - len_window)/t_shift)+1)
            for n in range(num_windows):
                section = range(n*t_shift, n * t_shift + len_window)
                self.windows.append(Window(each, section))
            
    def save(self, file_name):
        '''
            Saves 'self' to 'filename' with pickle
        '''
        filehandle = open(file_name, 'wb')
        pickle.dump(self, filehandle)
        filehandle.close
    
    def __copy__(self):
        copy = Activity_Split()
        copy.activity_count = self.activity_count
        copy.activity_lengths = self.activity_lengths
        copy.activities = self.activities
        # copy.filtered = self.filtered
        # copy.filter_on = self.filter_on
        # copy.windows = self.windows

        return copy

    def plot_activity_count(self, ax):
        '''
            Creates plot showing the total number of each activity in the 'self.activities' list
        '''
        counts = np.array(list(self.activity_count.values()))
        keys = np.array(list(self.activity_count.keys()))
        self.sorted_order = np.argsort(counts)[-1::-1]

        plt.bar(range(keys.shape[0]), counts[self.sorted_order], color="r", align="center")
        plt.xticks(range(len(keys)), keys[self.sorted_order], fontsize=20, rotation=45)
        ax.set_ylabel('# of occurences')
        ax.set_title('Frequency of Actions Across Dataset')

    def boxplots_timespan(self, ax):
        '''
            Creates boxplots showing the distribution of 'time spans' for each activity in self.activities
        '''
        keys = list(self.activity_lengths.keys())
        counts = list(self.activity_lengths.values())
        n = len(counts)

        keys_sorted = []        # Organize by frequency
        counts_sorted = []
        for idx in self.sorted_order:
            keys_sorted.append(keys[idx])
            counts_sorted.append(counts[idx])

        plt.boxplot(counts_sorted, positions=np.array(range(n)) * 2.0 - 0.4, sym='', widths=0.6, showfliers=True)

        plt.xticks(np.array(range(n)) * 2.0 - 0.4, keys_sorted, fontsize=20, rotation=45)
        ax.set_ylabel('time (min)', fontsize=15)
        ax.set_title('Time span for statistics per activity')
        # bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6)
        # bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6)


class Activity(object):
    '''
        This class stores relevant data for a single activity
    '''

    def __init__(self, row):
        self.start = row['start']
        self.end = row['end']
        self.span = self.end - self.start
        self.category = ACTIVITY_PREFIX[row['name'][:2]]
        self.name = row['name'][2:].replace('_', ' to ').capitalize()

    def grab_data(self, data):
        '''
            grabs data points from timepoints around a single activity
        '''
        keep_rows = (data.acceleration.index >= self.start) & (data.acceleration.index <= self.end)
        self.accel = data.acceleration[keep_rows]
        self.has_video = 0
        for video_src in data.videos_lst:
            keep_rows = (video_src.index >= self.start) & (video_src.index <= self.end)
            if sum(keep_rows):
                if self.has_video:
                    self.video = pd.concat([self.video, video_src[keep_rows]]).sort_index()
                else:
                    self.video = video_src[keep_rows]
                self.has_video += 1


class Window(object):
    '''
        Similar to Activity but for a window of the Activity... should be combined with Activity
    '''

    def __init__(self, activity, section):
        '''
            Sets attributes and calls to grab any video data
        '''
        self.accel = activity.accel.iloc[section].values
        self.t_index = activity.accel.iloc[section].index
        self.start = np.amin(self.t_index)
        self.end = np.amax(self.t_index)
        self.category = activity.category
        self.name = activity.name
        self.has_video = False
        if activity.has_video:
            self._grab_video_(activity)

    def _grab_video_(self, data):
        '''
            grabs video if exists in window span
        '''
        keep_rows = (data.video.index >= self.start) & (data.video.index <= self.end)
        if sum(keep_rows):
            self.video = data.video[keep_rows]
            self.has_video = True


if __name__ == "__main__":
    root_path = "/Users/caseyolson/Dropbox/My Mac (Caseys-MacBook-Pro.local)/Desktop/den-19/Capstone2/"
    data_path = root_path + "data/external/train/"
    meta_path = root_path + "data/external/metadata"

    lst = []                ###### pulling in all the raw train data... ######
    for n in range(1, 10):
        file = 100000 + n
        train_data = Data_Sequence(meta_path, data_path + str(file)[1:] + '/')
        train_data.load_data()
        lst.append(train_data)

    all_data = Activity_Split()

    for count, x in enumerate(lst):           ###### combine and separate all raw data into independent activity events... ######
        all_data.add_data(x)
        print(count)

    a_actions = ['Ascend', 'Descend', 'Jump', 'Loadwalk', 'Walk']           # Ambulation Activities
    p_actions = ['Bent', 'Kneel', 'Lie', 'Sit', 'Squat', 'Stand']           # Posture Activities
    t_actions = ['Bend', 'Kneel to stand', 'Lie to sit', 'Sit to lie', 'Sit to stand',
                 'Stand to kneel', 'Stand to sit', 'Straighten', 'Turn']    # Transition Activities
    actions = a_actions + p_actions + t_actions

               ###### filter activities list by time span and activity ######
    all_data.filter_data(actions, t_window=1, t_shift=0.5)

            #  filter_data(action_list, t_window=1, t_shift=0.5)

    # all_data.save('data/interim/all_data_1o5swindow_0o5shift.obj')

