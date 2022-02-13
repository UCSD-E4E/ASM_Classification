# Postprocessing Team

## Team Member
* Bruce

## Task Specification
We want a script that can do the following

### Input
Here are the relavent information needed for the Formating stage:
* A path to csv file containing the predited label
    * format: [(pic_file_name1, label1),(pic_file_name2, label2)...]
* A path to csv file containing the time-stamp mapping information
    * format: [(pic_file_name1, time_stamp1),(pic_file_name2, time_stamp1)...]

### Output
The output of the Formating stage should be 
* A path to a csv file containing the labels of all time stamps
    * format: [(time_stamp1, label1),(time_stamp2, label2)...]