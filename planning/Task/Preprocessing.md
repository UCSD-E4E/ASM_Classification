# Preprocessing Team

## Team Member
* Kirtan
* Anita

## Task Specification
We want a script that can do the following

### Input
Here are the relavent information needed for the preprocessing stage:
* A path to a csv? file that contains a list of video file name
    * the file name should be composed of {(device_id)/(starting_time_stamp)}.mp4
* A path to a csv? file that cotains a list of tuple
    * format: [(time_stamp1, label1),(time_stamp2, label2)...]


### Output
The output of the preprcessing stage should be 
* A set of images 
    * frames from the video set
* A csv file containing the label information
    * format: [(pic_file_name1, label1),(pic_file_name2, label2)...]
* A csv file containing the time-stamp mapping information
    * format: [(pic_file_name1, time_stamp1),(pic_file_name2, time_stamp1)...]
