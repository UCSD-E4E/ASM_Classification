import numpy as np
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
from AyeAyeDataset import AyeAyeDataset
import torch.optim as optim
import torch.nn as nn
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def main():
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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    # Load the Training Dataset
    dataset_train = AyeAyeDataset(root="../asm-nas/processed_data", data_annotations="../asm-nas/processed_data/csv/Train.csv", data_frames="./frames", transforms=train_transform)
    # Load the Validation Dataset
    dataset_validation = AyeAyeDataset(root="../asm-nas/processed_data", data_annotations="../asm-nas/processed_data/csv/Validation.csv", data_frames="./frames", transforms=val_transform)
    # Load the Test Dataset
    dataset_test = AyeAyeDataset(root="../asm-nas/processed_data", data_annotations="../asm-nas/processed_data/csv/Test.csv", data_frames="./frames", transforms=val_transform)

    # Create the Training Dataset into a Dataloader
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    # Create the Validation Dataset into a Dataloader
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=True)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device Intialized")

    # Training
    if args.load is None:
        # Using a Binary Cross Entropy Loss function
        criterion = nn.BCEWithLogitsLoss()
        # Using SGD Optimizer (Other options are Adam/RMSprop)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        valid_loss_min = np.Inf
        print("Start Training")
        for epoch in range(args.num_epochs):

            train_loss = 0.0
            valid_loss = 0.0
            batch_idx = 0
            # train the model
            model.train()
            for image, label, name in data_loader:
                if batch_idx % 50 == 0:
                    print("Training batch {}".format(batch_idx))
                if torch.cuda.is_available():
                    image, label = image.cuda(), label.cuda()
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
                batch_idx += 1

            # validate the model
            model.eval()
            for image, label, name in data_loader_validation:
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
                torch.save(model.state_dict(), './src/utils/Saved_models/PyTorch_Binary_Classifier.pth')
                valid_loss_min = valid_loss

        print('Finished Training')

    # Load an existing model
    else:
        print("Load model")
        model.load_state_dict(torch.load(args.load))

    # Prediction on Test Set

    # Dictionary of file name to classification
    class_dict = {}
    preds = []
    labels = []
    # Go through each image in directory
    model.eval()
    for image, label, name in data_loader_test:
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = model(image)
        rslt = (output) > 0.5  # Get result of prediction
        if rslt[0][0].item() is True:
            class_dict[name] = 1  # Store prediction with corresponging image name
            preds.append(1)
        else:
            class_dict[name] = 0
            preds.append(0)
        labels.append(label[0].item())

    # ADD IN CSV WRITING HERE USING 'class_dict'
    csv_file_name = '../asm-nas/processed_data/csv/prediction_results_torch.csv'
    class_dict_df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['Owl'])
    class_dict_df.to_csv(csv_file_name, index_label="Name")
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
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--load', type=str, default=None, help='path to saved model, skip training if not None')
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--no_finetuning', action='store_true', help='only update fc layer')
    args = parser.parse_args()
    main()