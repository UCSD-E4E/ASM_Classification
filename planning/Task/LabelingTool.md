# Labelling Tool Development

## Team Member


## Task Specification
To develop the dataset for training, we need a lebelling tool to efficiently 
label the input videos.
Preferably a UI tool. but the goal is to quickly label the video frames

### Input
* A list contatining the video file names that need to be labeled
    * format: [`video_file1`, `video_file2`, ... ]
    * the time info is embeded in the file names

### Output
The output of the Formating stage should be 
* A path to a csv file containing the labels of all time stamps
    * format: [(`time_stamp1, label1`),(`time_stamp2, label2`)...]
    * These time_stamps only marked when there is a change