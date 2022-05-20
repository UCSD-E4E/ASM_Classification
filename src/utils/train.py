import numpy as np
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
from AyeAyeDataset import AyeAyeDataset
from split_data import split
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)

    # How many classes
    output_dimensions = 1

    # Transformation to image
    # Need ToTensor and Normalize
    train_transform = transforms.Compose(
        [
            # Data augmentation, other augmentation methods
            # did not give better results
            transforms.RandomResizedCrop(100),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=.7, contrast=.3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(100),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    
    if args.no_data_split is False:
        train, test, val = split(args.pic_label, 0.2, 0.25, None, None)

    if args.load_n_test is None:
        if args.no_data_split:
            # Load the Training Dataset
            dataset_train = AyeAyeDataset(root=args.frame_root, data_annotations=args.train_csv, data_frames="./data_feb/frames", transforms=train_transform)
            # Load the Validation Dataset
            dataset_validation = AyeAyeDataset(root=args.frame_root, data_annotations=args.valid_csv, data_frames="./data_feb/frames", transforms=val_transform)
        else:
            dataset_train = AyeAyeDataset(root=args.frame_root, data_annotations_df=train, data_frames="./data_feb/frames", transforms=train_transform)
            dataset_validation = AyeAyeDataset(root=args.frame_root, data_annotations_df=val, data_frames="./data_feb/frames", transforms=val_transform)
        # Create the Training Dataset into a Dataloader
        data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        # Create the Validation Dataset into a Dataloader
        data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=True)

    if args.no_data_split:
        # Load the Test Dataset
        dataset_test = AyeAyeDataset(root=args.frame_root, data_annotations=args.test_csv, data_frames="./data_feb/frames", transforms=val_transform)
    else:
        dataset_test = AyeAyeDataset(root=args.frame_root, data_annotations_df=test, data_frames="./data_feb/frames", transforms=val_transform)
    # Create the Test Dataset into a Dataloader
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)

    # Load pretrained RESNET-18
    # Torchvision has other models we can experiment with if needed
    model = torchvision.models.resnet18(pretrained=True)

    # If specified not to finetune, then freeze the parameters in the model
    if args.no_finetuning:
        for param in model.parameters():
            param.requires_grad = False

    # Add fully connected layer and sigmoid function for final prediction
    model.fc = nn.Sequential(
        torch.nn.Linear(512, output_dimensions),
        torch.nn.Sigmoid()
    )

    # If cuda is available, will use that or then your computer's CPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"
    model.to(device)
    print("Device Intialized: {}".format(device))

    # Training
    if args.load_n_test is None:
        # Using a Binary Cross Entropy Loss function
        criterion = nn.BCEWithLogitsLoss()
        # Using SGD Optimizer (Other options are Adam/RMSprop)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        valid_loss_min = np.Inf
        print("Start Training")
        for epoch in range(args.num_epochs):
            print("====================== Epoch {}/{} ======================".format(epoch, args.num_epochs))
            train_loss = 0.0
            valid_loss = 0.0
            # train the model
            model.train()
            for image, label, name in tqdm(data_loader):
                if torch.cuda.is_available():
                    image, label = image.to(device), label.to(device)
                # Zero the gradients
                optimizer.zero_grad()
                # Predict the image
                output = model(image)
                # Change actual label into format criterion can read
                label = label.unsqueeze(1)
                label = label.float()
                # Find the Loss value
                loss = criterion(output, label)
                # Back-Propagate
                loss.backward()
                # Update model parameters
                optimizer.step()
                train_loss += loss.item()

            # validate the model
            model.eval()
            for image, label, name in tqdm(data_loader_validation):
                if torch.cuda.is_available():
                    image, label = image.cuda(), label.cuda()
                # Predict the image
                output = model(image)
                # Change actual label into format criterion can read
                label = label.unsqueeze(1)
                label = label.float()
                # Find the Loss value
                loss = criterion(output, label)
                valid_loss += loss.item()

            train_loss = train_loss/len(data_loader)
            valid_loss = valid_loss/len(data_loader_validation)
            
            #print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")
            print("Epoch: {} \tTraining Loss: {} \tValidation Loss: {}".format(epoch, train_loss, valid_loss))

            # Save new model if there is a lower validation loss
            if valid_loss <= valid_loss_min:
                print("Validation loss decreased (",valid_loss_min, " --> ",valid_loss, ").  Saving model")
                torch.save(model.state_dict(), os.path.join(args.check_point_path, args.model_name))
                valid_loss_min = valid_loss

        print('Finished Training')

    # Load an existing model
    else:
        print("Load model")
        model.load_state_dict(torch.load(args.load_n_test))

    # Prediction on Test Set

    # Dictionary of file name to classification
    class_dict = {}
    preds = []
    labels = []
    print('Start Testing')
    # Go through each image in directory
    model.eval()
    for image, label, name in tqdm(data_loader_test):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = model(image)
        rslt = (output) > 0.5  # Get result of prediction
        if rslt[0][0].item() is True:
            class_dict[name[0]] = 1  # Store prediction with corresponging image name
            preds.append(1)
        else:
            class_dict[name[0]] = 0
            preds.append(0)
        labels.append(label[0].item())

    # ADD IN CSV WRITING HERE USING 'class_dict'
    class_dict_df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['label'])
    class_dict_df.to_csv(args.output_csv_path, index_label="pic_name")
    print('Confusion Matrix: ')
    #                    predicted no:        predicted yes:
    # actual no:         Tn                      Fp
    # actual yes:        Fn                      Tp
    print(confusion_matrix(labels, preds))
    print('')
    print('Accuracy score: ')
    #      Tp+Tn
    #   ------------
    #   Tp+Tn+Fp+Fn
    print(accuracy_score(labels, preds))

    print("")
    print(classification_report(labels, preds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--no_finetuning', action='store_true', help='only update fc layer')
    parser.add_argument("--device", type=str, default = 'cuda:0')

    parser.add_argument('--model_name', type=str, default='PyTorch_Binary_Classifier.pth', help='Name of the model to be saved, eg. PyTorch_Binary_Classifier.pth')
    parser.add_argument('--load_n_test', type=str, default=None, help='path to saved model, skip training if not None')
    parser.add_argument("--check_point_path", type=str)

    parser.add_argument("--pic_label", type=str, default = '/mnt/aye-aye-sleep-monitoring/demo_data/may/pic_label.csv')
    parser.add_argument("--no_data_split", action='store_true', help='disable train test split and use existing csv files')
    parser.add_argument("--train_csv", type=str, default = '/mnt/aye-aye-sleep-monitoring/processed_data/csv/Train.csv')
    parser.add_argument("--valid_csv", type=str, default = '/mnt/aye-aye-sleep-monitoring/processed_data/csv/Validation.csv')
    parser.add_argument("--test_csv", type=str, default = '/mnt/aye-aye-sleep-monitoring/processed_data/csv/Test.csv')
    parser.add_argument("--frame_root", type=str, default = '/mnt/aye-aye-sleep-monitoring/processed_data')
    parser.add_argument("--output_csv_path", type=str)
    
    args = parser.parse_args()
    main(args)