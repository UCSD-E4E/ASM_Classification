import argparse
import os
import yaml

import pandas as pd

def formatting(path_label: str, path_time_stamp: str, config) -> str:
    '''
    path_label: A path to csv file containing the predicted label
        format: [(pic_file_name1, label1),...]
    path_time_stamp: A path to csv file containing the time-stamp mapping information
        format: [(pic_file_name1, time_stamp1),...]
    return: A path to the output csv file containing the labels of all time stamps
        format: [(time_stamp1, label1),...]
    '''
    
    out_format = config['format']

    pic_label_df = pd.read_csv(path_label, names = ['pic_file_name', 'label'])
    pic_time_df = pd.read_csv(path_time_stamp, names = ['pic_file_name', 'time_stamp'])
    
    if out_format == 'time_label':
        pic_label_df = pic_label_df.set_index('pic_file_name')
        pic_label_df = pic_label_df.reindex(index=pic_time_df['pic_file_name'])
        pic_label_df = pic_label_df.reset_index()
        labels = pic_label_df['label']
        time_stamps = pic_time_df['time_stamp']
        time_label = {'time_stamp': time_stamps, 'label': labels}
        time_label_df = pd.DataFrame(time_label)
        path_label_time = './postprocessing_output.csv'
        time_label_df.to_csv(path_label_time, index=False)
    else:
        raise NotImplementedError('Output format {} is not implemented or defined'.format(out_format))
    return path_label_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='A path to the format configuration file.')
    parser.add_argument('--path_label', help='A path to csv file containing the predicted label.')
    parser.add_argument('--path_time', help='A path to csv file containing the time-stamp mapping information.')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.BaseLoader)
    path_label_time = formatting(args.path_label, args.path_time, config)

if __name__ == '__main__':
    main()