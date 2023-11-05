import argparse
import json
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

# Define command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

# Define transforms for the training, validation, and testing sets
def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = valid_transform
    return train_transform, valid_transform, test_transform

# Load the datasets with ImageFolder
def get_dataloaders(data_dir, train_transforms, valid_transform, test_transform):
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transform)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return trainloader, validloader, testloader, train_data

# Function to create the model
def build_model(arch='vgg16', hidden_units=512, output_size=102):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    return model

# Function to train the model
def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device):
    steps = 0
    running_loss = 0
    print("Model training...")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("Training is done")

# Function to validate the model on test data
def test_validation(model, testloader, criterion, device):

    print('Now Measuring Test accuracy...')
    test_loss = 0
    test_accuracy = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test loss: {test_loss/len(testloader):.3f}, "
          f"Test accuracy: {test_accuracy/len(testloader):.3f}")

# Function to save the checkpoint
def save_checkpoint(model, optimizer, train_data, args, file_path):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': args.arch,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print("Checkpoint saved!")

# Main program
def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    train_transform, valid_transform, test_transform = get_data_transforms()
    trainloader, validloader, testloader, train_data = get_dataloaders(args.data_dir, train_transform, valid_transform, test_transform)
    model = build_model(arch=args.arch, hidden_units=args.hidden_units, output_size=102)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device)
    train_model(model, trainloader, validloader, args.epochs, 20, criterion, optimizer, device)
    test_validation(model, testloader, criterion, device)  # Validate on the test set
    save_checkpoint(model, optimizer, train_data, args, args.save_dir + '/checkpoint.pth')

if __name__ == '__main__':
    main()
