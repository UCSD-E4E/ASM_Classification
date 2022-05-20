import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def split(data_set, test_size, val_size,  file_name, save_directory = None):
    """
    Function that split a given dataset into train, test, validation set.
    data_set :  path to the dataset
    test_size: proportion of data_set as test_set
    val_size : proportion of train_set as val_set
    save_directory : directory to save the splitted datas if needed
    file_name : saved_filename in the format of {file_name}_train/valie/test.csv
    """
    df = pd.read_csv(data_set)
    train, test = train_test_split(df, test_size=test_size, random_state=1)
    train, val = train_test_split(train, test_size=val_size, random_state=1)
    if save_directory != None:
        dir = os.path.isdir(save_directory)
        if dir == False:
            raise Exception('Directory does not exist')
        train_file = os.path.join(save_directory, '{}_train.csv'.format(file_name))
        valid_file = os.path.join(save_directory, '{}_valid.csv'.format(file_name))
        test_file = os.path.join(save_directory,  '{}_test.csv'.format(file_name))
        train.to_csv(train_file)
        val.to_csv(valid_file)
        test.to_csv(test_file)

    return train, test, val

# testing code
# if __name__ == '__main__':
#     split('/home/yul080/ASM_Classification/data_csv/april_data/april_pic_label.csv',0.2,0.25,'april',save_directory = '/home/yul080/ASM_Classification/nonexistingS')

