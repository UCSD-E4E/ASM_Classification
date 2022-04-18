import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import argparse
import os
from inference import inference_func

def on_created(event):
    csv_ = os.path.join(args.output_csv_folder, (event.src_path.split('/')[-1].split('.mp4')[0]+'.csv'))
    print(event.src_path)
    time.sleep(5)
    inference_func(args.load_n_test, event.src_path,args.seed,csv_)
    print(csv_)
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
    # return event.src_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--load_n_test', type=str, default=None, help='path to saved model, skip training if not None')
    parser.add_argument("--folder", type=str, default = '/home/burrowingowl/asm-nas/processed_data')
    parser.add_argument("--output_csv_folder", type=str, default = '/home/burrowingowl/ASM_Classification/data/prediction_results_torch.csv')

    args = parser.parse_args()
    patterns = ["*"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_modified = on_modified
    my_event_handler.on_created = on_created
    my_event_handler.on_moved = on_moved
    path = args.folder
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    try:
        while True:
           time.sleep(5)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
