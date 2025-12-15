# basic data handling
import numpy as np
import pandas as pd
import re

# to pull the files from dropbox directly
import dropbox
from dropbox.files import SharedLink
import tempfile

# for managing .mat files
import scipy.io
import matlab.engine
import io

# for handling access codes for the dropbox api
from dotenv import load_dotenv
import os

# basic file handling
from os import listdir
from os.path import isfile, join

def parse_mat_from_dropbox(dbx, eng, file, url, table2parse='data'):
    '''
    opens a matlab 5.0 mat file from dropbox, loads the file into a temporary file, then uses the matlab engine to modify
    a Matlab table in that temp file such that it can be read by scipy.io.loadmat (otherwise it will be read as a MatlabOpaque object)

    Parameters:
    - dbx: a Dropbox object
    - eng: a MatlabEngine object
    - file: the file path of the .mat file located in the shared dropbox
    - url: the url of the shared dropbox
    - table2parse: the name of the Matlab table that we want to convert to a python-readable struct

    Returns:
    - pd.Dataframe of parsed data, dictionary containing all variables in the .mat file (including the original unparsed table)
    '''
    # download file into memory
    _, res = dbx.sharing_get_shared_link_file(path=file, url=url)

    # access content directly
    file_bytes = res.content # raw bytes

    # store data in a temporary mat file so we can access it with the matlab engine
    with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        eng.eval(f"load('{tmp.name}')", nargout=0)
        eng.eval(f"parsed_data = table2struct({table2parse},'ToScalar',true); save('{tmp.name}','parsed_data', '-append')", nargout=0)

        # convert to pandas dataframe
        m = loadmat(tmp.name)
        return pd.DataFrame(m['parsed_data']), m

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# main
def convert_to_pkl():
    '''
    Pulls directly from the Camargo dataset dropbox and outputs a set of pkl files.
    '''
    # need to convert original matlab tables into python readable format
    # do this by running matlab code from matlab's python api
    eng = matlab.engine.start_matlab()

    # get access token from env file
    load_dotenv()
    access_token = os.getenv("DROPBOX_TOKEN")
    dbx = dropbox.Dropbox(access_token)
    link = SharedLink(url='https://www.dropbox.com/scl/fo/i2ee8z2nxjwkf3mdocw63/AEuNRES_bD2n_Yt0xYzjHQ4?rlkey=eiqcgo6gyvf4bnfgduq419yyh&e=2&st=x2v83q3j&dl=0')

    # locomotion modes
    modes = ['levelground', 'ramp', 'stair', 'treadmill']

    # get all participants
    res = dbx.files_list_folder(path="", shared_link=link)
    participants = [e.name for e in res.entries if 'AB' in e.name]

    # get all files currently in the data folder
    data_path = 'data/01_dataset_as_pkl/'
    existing_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # loop over participants
    for p in participants:
        # get folder containing data
        pattern = re.compile(r"^\w+_\w+_\w+$")  # the folder with data in it is a date of structure xx_xx_xx, ignore any other folders/files
        res = dbx.files_list_folder(path=f"/{p}/", shared_link=link)
        date_folder = [e.name for e in res.entries if pattern.match(e.name)]
        
        tmp_path = f"/{p}/{date_folder[0]}/"
        for m in modes:
            tmp_path_mode = tmp_path + m + "/"

            # get list of all conditions files
            cond_files = tmp_path_mode + 'conditions/'
            emg_files = tmp_path_mode + 'emg/'
            imu_files = tmp_path_mode + 'imu/'
            mar_files = tmp_path_mode + 'markers/'

            # get list of file names, the names are identical for each folder (conditions, emg, etc)
            res = dbx.files_list_folder(path=cond_files, shared_link=link)
            file_names = [e.name for e in res.entries if '.mat' in e.name]

            for file in file_names:
                print(f'{p}, {m}, {file}')

                # all files will be named this (with some prefix)
                pkl_name = str.split(file, '.')[0] + '.pkl'

                if (p + '_condition_' + pkl_name) in existing_files:
                    # print('data already saved')
                    continue

                # initialize conditions dataframe
                cond_df = pd.DataFrame(
                    columns=['timeStamp', 'Label', 'Speed',                # if treadmill data, then we use speed (integer) instead of label (str)
                             'leadingLegStart', 'leadingLegStop',           # used for level ground data
                             'transLegAscent[0]', 'transLegAscent[1]',     # used for ramp and stair data
                             'transLegDescent[0]', 'transLegDescent[1]',   # used for ramp and stair data
                             'rampIncline', 'stairHeight'])                # for ramp and stair data

                # get the condition file
                if m == 'treadmill':        # labels table has a different name for treadmill data
                    table2parse = 'speed'
                else:
                    table2parse = 'labels'
                tmp_cond_df, tmp_cond_dict = parse_mat_from_dropbox(dbx, eng, file=cond_files+file, url=link.url, table2parse=table2parse)
                
                # get labels
                cond_df['timeStamp'] = tmp_cond_df['Header']
                if m == 'treadmill':
                    cond_df['Speed'] = tmp_cond_df['Speed']
                else:
                    cond_df['Label'] = tmp_cond_df['Label']

                # get other metadata
                # this is pretty ugly but each locomotion type has a condition file containing different variables
                variables = ['leadingLegStart', 'leadingLegStop', 'transLegAscent', 'transLegDescent', 'rampIncline', 'stairHeight']
                for var in variables:
                    if var in ['transLegAscent', 'transLegDescent']:    # these are arrays
                        try:
                            cond_df[f'{var}[0]'] = [tmp_cond_dict[f'{var}'][0]] * len(tmp_cond_df['Header'])
                            cond_df[f'{var}[1]'] = [tmp_cond_dict[f'{var}'][1]] * len(tmp_cond_df['Header'])
                        except:
                            cond_df[f'{var}[0]'] = [None] * len(tmp_cond_df['Header'])
                            cond_df[f'{var}[1]'] = [None] * len(tmp_cond_df['Header'])
                            pass
                    else:
                        try:
                            cond_df[var] = [tmp_cond_dict[var][0]] * len(tmp_cond_df['Header'])
                        except:
                            cond_df[var] = [None] * len(tmp_cond_df['Header'])
                            pass

                # save trial metadata
                pkl_path_c = 'data/01_dataset_as_pkl/' + p + '_condition_' + pkl_name
                cond_df.to_pickle(pkl_path_c)

                # get the emg data
                emg_df, _ = parse_mat_from_dropbox(dbx, eng, file=emg_files+file, url=link.url)
                pkl_path_e = 'data/01_dataset_as_pkl/' + p + '_emg_' + pkl_name
                emg_df.to_pickle(pkl_path_e)

                # imu file
                imu_df, _ = parse_mat_from_dropbox(dbx, eng, file=imu_files+file, url=link.url)
                pkl_path_i = 'data/01_dataset_as_pkl/' + p + '_imu_' + pkl_name
                imu_df.to_pickle(pkl_path_i)

                # marker file
                marker_df, _ = parse_mat_from_dropbox(dbx, eng, file=mar_files+file, url=link.url)
                pkl_path_m = 'data/01_dataset_as_pkl/' + p + '_marker_' + pkl_name
                marker_df.to_pickle(pkl_path_m)
                if p == 'AB08' and file == 'ramp_1_l_01_04.mat':
                    print(marker_df)
                    break
   
    # close matlab engine
    eng.quit()

if __name__ == "__main__":
    convert_to_pkl()