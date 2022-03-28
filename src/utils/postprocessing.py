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
    OUTPUT_CSV = '/home/burrowingowl/ASM_Classification/data/postprocessing_output_change.csv'
    out_format = config['format']

    pic_label_df = pd.read_csv(path_label)
    pic_time_df = pd.read_csv(path_time_stamp)
    pic_label_df.drop_duplicates(subset=['pic_name'], inplace=True)
    pic_time_df.drop_duplicates(subset=['pic_name'], inplace=True)

    #pic_label_df = pic_label_df.set_index('pic_name')
    #pic_label_df = pic_label_df.reindex(index=pic_time_df['pic_name'])
    #pic_label_df = pic_label_df.dropna()
    #pic_label_df = pic_label_df.reset_index()
    pic_time_df = pic_time_df.set_index('pic_name')
    pic_time_df = pic_time_df.reindex(index=pic_label_df['pic_name'])
    pic_time_df = pic_time_df.dropna()
    pic_time_df = pic_time_df.reset_index()
    labels = pic_label_df['label']
    time_stamps = pic_time_df['timestamp']
    time_label = {'timestamp': time_stamps, 'label': labels}
    time_label_df = pd.DataFrame(time_label).dropna()
    time_label_df.sort_values(by='timestamp', inplace=True)
    time_label_df = time_label_df[['timestamp', 'label']]
    
    if out_format == 'time_label':
        path_label_time = './data/postprocessing_output.csv'
        time_label_df.to_csv(OUTPUT_CSV, index=False)
    elif out_format == 'time_label_change':
        time_label_df['no_change'] = time_label_df.label.eq(time_label_df.label.shift())
        time_label_df = time_label_df.loc[time_label_df['no_change'] == False]
        time_label_df.drop(['no_change'], axis=1, inplace=True)
        path_label_time = './data/postprocessing_output_change.csv'
        time_label_df.to_csv(OUTPUT_CSV, index=False)
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