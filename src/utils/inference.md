# Running the inference script

## Script 
``inference_main.py``
* I/P: Inputs are passed as args to the function 
    * seed --> random seed
    * load_n_test --> path to saved model
    * folder --> path to the folder to be watched for new videos
    * output_csv_folder --> Path to the output folder where the csv result for each video will be stored 
* O/P: csv for each new video containing timestamp and correpsonding prediction 

* Example run:\
``python  inference_main.py --seed 0 --load_n_test saved_model/checkpoint.pth --folder path_to_videos --output_csv_folder path_to_csv_folder``

## Testing
To test the script keep following points in mind:
1. Uncomment the parts in inference_main.py mentioned as 'UNCOMMENT DURING TESTING'
2. Copy a file to the folder passed as argument to --folder in the script 
3. You will know the file has been detected if the window starts outputing the folder path and other specifics 
4. If after pt 2. expected output in pt.3 is not observed please perform same copy operation again (NOTE: Do not remove the file, run the same cp command with same file again)

If facing any issues during testing please reach out to Anisha Pal on slack.
