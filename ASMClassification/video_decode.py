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
    # pdb.set_trace()
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

def get_frame(video):
    '''
        Reading frames and calculating the timestamp for each frame
        i/p: path to video file, path to root directory 
        o/p: list containg file name of each frame and timestamp, format: [(device_id/start_timestamp_frame_x.jpg,timestamp),...]
    '''
    # pdb.set_trace()
    list_pic_time = []
    list_frames = []
    start = video.split('/')[-1].split('.mp4')[0]
    vidcap = cv2.VideoCapture(video)
    
    #Getting the fps for a video, assuming fps is constant throughout the video (general case)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()

    if not success:
        sys.exit("Reading Video file failed")
    count = 0
    while success:

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
        list_pic_time.append(timestamp)
        list_frames.append(image)
        success,image = vidcap.read()
        count += 1
    return list_pic_time, list_frames


def fetch_frames(video_file):
    '''
        fetch_frames splits the video into frames and returns two lists containing the frames and corresponding timestamps
    '''    
    timestamp,frame = get_frame(video_file)
    return frame,timestamp

