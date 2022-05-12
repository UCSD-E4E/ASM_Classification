'''
    Creating a watcher which watches the folder where videos are getting stored for any new b=videos. As a new video is detected in teh folder it is passed through
    the inference pipeline. This activity repeats everytime a new video is created. Keep this script running in a tmux session.

    I/P: Inputs are passed as args to the function 
         seed --> random seed
         load_n_test --> path to saved model
         folder --> path to the folder to be watched for new videos
         output_csv_folder --> Path to the output folder where the csv result for each video will be stored 
    O/P: csv for each new video containing timestamp and correpsonding prediction 

'''

import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import argparse
import os
from inference import inference_func


def on_created(event):
    '''
        Function is called whenever a new video is detected in the folder being watched
    '''

    csv_ = os.path.join(args.output_csv_folder, (event.src_path.split('/')[-1].split('.mp4')[0]+'.csv'))
    print(event.src_path)

    #Waiting for 5 seconds for video file to finish saving before starting inference script
    time.sleep(5) 
    inference_func(args.load_n_test, event.src_path,args.seed,csv_)
    print(csv_)

'''
Uncomment only during testing the script 

def on_moved(event):
    csv_ = os.path.join(args.output_csv_folder, (event.src_path.split('/')[-1].split('.mp4')[0]+'.csv'))
    print(event.src_path)
    time.sleep(5)
    inference_func(args.load_n_test, event.src_path,args.seed,csv_)
    print(csv_)
def on_modified(event):
    csv_ = os.path.join(args.output_csv_folder, (event.src_path.split('/')[-1].split('.mp4')[0]+'.csv'))
    print(event.src_path)
    time.sleep(5)
    inference_func(args.load_n_test, event.src_path,args.seed,csv_)
    print(csv_)
    return event.src_path
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--load_n_test', type=str, default=None, help='path to saved model')
    parser.add_argument("--folder", type=str, default = '/home/burrowingowl/asm-nas/processed_data',help='path to the folder to be watched for new videos')
    parser.add_argument("--output_csv_folder", type=str, default = ' Path to the output folder where the csv result for each video will be stored ')

    args = parser.parse_args()

    '''
    “ignore_patterns” variable contains the patterns that we don’t want to handle, 
    “ignore_directories” is just a boolean that we can set to True if we want to be run script just for regular files (not directories)
    “case_sensitive” variable is just another boolean that, if set to “True”, made the patterns we previously introduced “case sensitive”
    '''
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True

    '''
        The event handler is the object that will be notified when something happen on the filesystem we are monitoring.
    '''
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_created = on_created
   
    '''
    Uncomment only during testing the script 

    my_event_handler.on_modified = on_modified
    my_event_handler.on_moved = on_moved
    '''
    path = args.folder

    '''
        Monitor our filesystem, looking for changes that will be handled by the event handler.
        go_recursively - if enabled will go through changes in sub directories as well
    '''
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    '''
         Starting the observer thread
    '''
    try:
        while True:
           time.sleep(5)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
