import pandas as pd 
import sys
import cv2
from tqdm import tqdm
import multiprocessing
import os
from pytz import utc, timezone
import argparse
import pdb
from functools import partial
import datetime

def convert_pst_time_to_utc_time(temp_time):
    '''
        Convert PST to UTC
    '''
    pacific_timezone = timezone('US/Pacific')
    pst_time = pacific_timezone.localize(temp_time, is_dst=None)
    assert pst_time.tzinfo is not None
    assert pst_time.tzinfo.utcoffset(pst_time) is not None
    utc_time = pst_time.astimezone(timezone('utc'))
    return utc_time

def calc_time(start,calc):
    '''
        Add time from each frame to the starting timestamp 
        i/p: starting timestamp, relative frame timestamp
        o/p: global frame timestamp in utc
    '''
    try:
        a = start.microsecond*pow(10,-6)+start.second + start.minute*60 + start.hour*60*60 + start.day*60*60*24
    except:
        a = start.second + start.minute*60 + start.hour*60*60 + start.day*60*60*24  
    a_delta = datetime.timedelta(0,a)
    b_delta = datetime.timedelta(0,calc)
    c = str(a_delta + b_delta)
    try:
        c = datetime.datetime.strptime(c, "%d days, %H:%M:%S.%f")
    except:
        c = datetime.datetime.strptime(c, "%d days, %H:%M:%S")
    date_sum = datetime.datetime(start.year,start.month,c.day,c.hour,c.minute,c.second,c.microsecond)
    date_final = date_sum.isoformat()
    return date_final

def get_frame(video,root_path,save_path):
    '''
        Reading frames and calculating the timestamp for each frame
        i/p: path to video file, path to root directory 
        o/p: list containg file name of each frame and timestamp, format: [(device_id/start_timestamp_frame_x.jpg,timestamp),...]
    '''

    list_pic_time = []
    start = video.split('/')[-1].split('.mp4')[0]
    vidcap = cv2.VideoCapture(os.path.join(root_path,video))
    
    #Getting the fps for a video, assuming fps is constant throughout the video (general case)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()

    if not success:
        sys.exit("Reading Video file failed")
        
    pic_name = os.path.join(save_path, 'frames',video.split('/')[-2], video.split('/')[-1].split('.mp4')[0])
    count = 0
    while success:
        file_name = pic_name + "_frame_%d.jpg" % count

        #Calculating the timestamp for each frame
        try:
            start1 = datetime.datetime.strptime(start, "%Y.%m.%d.%H.%M.%S.%f")
        except:
            start1 = datetime.datetime.strptime(start, "%Y.%m.%d.%H.%M.%S")
        start1 = convert_pst_time_to_utc_time(start1)
        calc = float(count/fps)
        timestamp = calc_time(start1,calc)
        # timestamp = start + 1000*float(count)/fps
        # timestamp = start + vidcap.get(cv2.CAP_PROP_POS_MSEC)
        list_pic_time.append([file_name.split('/')[-2]+'/'+file_name.split('/')[-1],timestamp])
        cv2.imwrite(file_name, image)
        success,image = vidcap.read()
        count += 1
    return list_pic_time

def get_dir(video_file,root_path):
    '''
        Getting the base directory for each frame
        i/p: path to video file, path to root directory 
        o/p: base directory for each frame
    '''
    # TODO: Video_files are file name, not including path
    # the full path should be args.root_path/video_filename
    return os.path.join(root_path,'frames',video_file.split('/')[-2])

def match_pic_label(df_pic_time, df_label):
    '''
        Perform a linear search through the two sorted dataframes by timestamp value to assign label to each frame.
        i/p: dataframe containing pic,timestamp information and dataframe containing label,timestamp information
        o/p: dataframe containing pic,label information
    '''
    # pdb.set_trace()
    pic_label_list = []
    df_label = df_label.sort_values('timestamp').reset_index(drop=True)
    df_pic_time = df_pic_time.sort_values('timestamp').reset_index(drop=True)
    start = 0
    for i, row in tqdm(df_pic_time.iterrows(),total = len(df_pic_time)):
        try:
            pic_time = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
        except:
            pic_time = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%dT%H:%M:%S")

        if (start+1 >= len(df_label)):
            pic_label_list.append([row['pic_name'],df_label['label'][start]])
        else:
            try: 
                label_time = datetime.datetime.strptime(df_label['timestamp'][start+1], "%Y-%m-%dT%H:%M:%S.%f")
            except:
                label_time = datetime.datetime.strptime(df_label['timestamp'][start+1], "%Y-%m-%dT%H:%M:%S")
            if(pic_time<label_time):
                pic_label_list.append([row['pic_name'],df_label['label'][start]])
            else:
                pic_label_list.append([row['pic_name'],df_label['label'][start+1]])
                start = start+1
    return pd.DataFrame(pic_label_list, columns =['pic_name', 'label'])


def main():
    parser = argparse.ArgumentParser(description="Preprocessing -> frame extaction and label matching")
    parser.add_argument('--video_csv', type=str, required = True,help='path to csv file that contains a list of video file name')
    parser.add_argument('--label_csv', type=str,required = True, help='path to csv file containing label and corresponding timestamp information')
    parser.add_argument('--root_path', type=str, required = True, help='path to root directory where videos are saved')
    parser.add_argument('--save_path', type=str, required = True, help='path to directory where frames will be saved')
    parser.add_argument('--mp', action='store_true',help='enable multiprocessing')
    args = parser.parse_args()

    if not(os.path.exists(os.path.join(args.save_path))):
        os.makedirs(os.path.join(args.save_path))    
    if not(os.path.exists(os.path.join(args.save_path,'frames'))):
        os.makedirs(os.path.join(args.save_path,'frames'))
    
    df_video = pd.read_csv(args.video_csv)
    df_label = pd.read_csv(args.label_csv)
    videos = df_video["video"].tolist()
    pic_time_list = []


    #Creating the directory structure for video frames

    dir_name = list(set(map(partial(get_dir,root_path=args.save_path), videos)))
    for i in dir_name:
        if not(os.path.exists(i)):
            os.makedirs(i)

    
 
    if(args.mp):
        pool = multiprocessing.Pool(os.cpu_count())
        for i in tqdm(pool.imap_unordered(partial(get_frame,root_path = args.root_path,save_path=args.save_path), videos), total = len(videos)):
            pic_time_list.extend(i)
    else:
        for i, video in tqdm(enumerate(videos),total=len(videos)):
            frame = get_frame(video,args.root_path,args.save_path)
            pic_time_list.extend(frame)
    # pdb.set_trace()
   
    df_pic_time =pd.DataFrame(pic_time_list, columns =['pic_name', 'timestamp']) 
    
    #Output csv format: pic_name, timestamp
    df_pic_time.to_csv(os.path.dirname(args.video_csv)+'/pic_timestamp.csv',index = False)

    #Matching each frame with corresponding label
    df_pic_label = match_pic_label(df_pic_time, df_label)

    #Output csv format: pic_name, label
    df_pic_label.to_csv(os.path.dirname(args.video_csv)+'/pic_label.csv',index = False)

if __name__ == "__main__":
   main()
