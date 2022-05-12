import numpy as np
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
from AyeAyeDataset import AyeAyeDatasetInference
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def inference_func(load_n_test, video_file, seed, output_csv_path):
    # Set random seed
    torch.manual_seed(seed)

    # How many classes
    output_dimensions = 1

    # Load the Test Dataset
    test_transform = transforms.Compose([transforms.CenterCrop(100),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    '''
        AyeAyeDatasetInference splits the video file into frames and returns frames and corresponding timestamps 
    '''
    dataset_test = AyeAyeDatasetInference(video_file=video_file, transforms=test_transform)

    # Load the Test Dataset into a Dataloader
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)
    print('TEST', len(data_loader_test))

    # Load pretrained RESNET-18
    # Torchvision has other models we can experiment with if needed
    model = torchvision.models.resnet18(pretrained=True)


    # Add fully connected layer and sigmoid function for final prediction
    model.fc = nn.Sequential(
        torch.nn.Linear(512, output_dimensions),
        torch.nn.Sigmoid()
    )

    # If cuda is available, will use that or then your computer's CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device Intialized: {}".format(device))


    # # Load an existing model
    print("Load model")
    model.load_state_dict(torch.load(load_n_test))

    # Prediction on Test Set

    # Dictionary of file name to classification
    class_dict = {}
    # preds = []
    print('Start Testing')
    # Go through each image in directory
    model.eval()
    for image, name in tqdm(data_loader_test):
        if torch.cuda.is_available():
            image = image.cuda()
        output = model(image)
        rslt = (output) > 0.5  # Get result of prediction
        if rslt[0][0].item() is True:
            class_dict[name[0]] = 1  # Store prediction with corresponging image name
            # preds.append(1)
        else:
            class_dict[name[0]] = 0
            # preds.append(0)

    # ADD IN CSV WRITING HERE USING 'class_dict'
    class_dict_df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['label'])
    class_dict_df.to_csv(output_csv_path, index_label="pic_name")