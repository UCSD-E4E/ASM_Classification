# Running the train script

## Command line arguments
- Hyperparameters and training:
    - ``num_epochs``: Number of epochs for training.
    - ``batch_size``: Number of data samples to run in one iteration.
    - ``seed``: Random seed.
    - ``no_finetuning``: Disable parameter updates in the CNN. Only train the fc layer.
    - ``device``: The device used to train and evaluate the model (e.g., cuda:0).
- Model:
    - ``model_name``: Name of the model to be saved (e.g., ``PyTorch_Binary_Classifier.pth``).
    - ``load_n_test``: Path to saved model, skip training if not None.
    - ``check_point_path``: Path to the directory where the model will be saved.
- Data:
    - ``pic_label``: Path to the csv file where "pic_name" and "label" pairs are saved, i.e., the output file from ``frames_v1.py``. This file is used for train test split.
    - ``no_data_split``: Disable train test split and use existing train, validation, and test csv files.
    - ``train_csv``: Path to an existing csv file containing the training data.
    - ``valid_csv``: Path to an existing csv file containing the validation data.
    - ``test_csv``: Path to an existing csv file containing the test data.
    - ``frame_root``: Path to the direction where frames are saved.
    - ``output_csv_path``: Path to the csv file where the predictions on the test set will be saved.

## Examples
- Train a model with 2 epochs, batch size of 64, 1 gpu with a pic_label file.
``python3 train.py --num_epochs 2 --batch_size 64 --device cuda:0 --model_name MODEL_NAME --check_point_path PATH_TO_CHECKPOINTS --pic_label PATH_TO_PIC_LABEL_FILE --frame_root PATH_TO_DATA_FOLDER --output_csv_path OUTPUT_CSV_FILE_NAME``

- Train a model with 2 epochs, batch size of 64, 1 gpu, and a different random seed. Use existing train, validation, and test files.
``python3 train.py --num_epochs 2 --batch_size 64 --seed 100 --device cuda:0 --model_name MODEL_NAME --check_point_path PATH_TO_CHECKPOINTS --no_data_split --train_csv PATH_TO_TRAIN_CSV_FILE --valid_csv PATH_TO_VALIDATION_CSV_FILE --test_csv PATH_TO_TEST_CSV_FILE --frame_root PATH_TO_DATA_FOLDER --output_csv_path OUTPUT_CSV_FILE_NAME``

- Evaluate a model without training.
``python3 train.py --device cuda:0 --load_n_test PATH_TO_SAVED_MODEL_FILE --pic_label PATH_TO_PIC_LABEL_FILE --frame_root PATH_TO_DATA_FOLDER --output_csv_path OUTPUT_CSV_FILE_NAME``

## Interpreting the results
- Accuracy score = # samples correctly predicted / # samples
- Classification_report:
    - precision = true positive / (true positive + false positive)
    - recall = true positive / (true positive + false negative)
    - f1-score = 2*(Recall * Precision) / (Recall + Precision)
    - support = # positive samples from each class
