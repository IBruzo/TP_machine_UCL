import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd


class CustomDataset(Dataset):
    
    # Constructor for the dataset
    def __init__(self, images, images_directory, target=None, transform=None):
        # Initialize the dataset with the provided data and transformation options
        self.images = images #List of image filenames
        self.images_directory = images_directory # Directory where images are located
        self.target = target # Optional list of target labels
        
        # If no data transformation is provided, create a default transformation
        if transform is None:
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
                                            transforms.ToTensor(), # Convert the image to a PyTorch tensor
                                            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize the image data 
                                           ])
            
        self.transform = transform # Store the data transformation for later use

    # Define the length of the dataset (number of data samples)
    def __len__(self):
        return len(self.images)
    
    # Retrieve a specific data sample by its index
    def __getitem__(self, idx):
        # Construct the full path to the image file
        image_path = os.path.join(self.images_directory, self.images[idx])
        # Open the image using the PIL library
        image = Image.open(image_path)
        
        # Apply the data transformation if it exists
        if self.transform:
            image = self.transform(image)
        
        # If target labels are provided, return both the image and the corresponding label esle return only the image
        if self.target is not None:
            target = self.target[idx]
            return image, target
        else:
            return image

# Define a custom CNN class (in this class, we define the network architecture)  
class SimpleCNN(nn.Module):
    
    # Constructor for the CNN 
    def __init__(self, n_features):
        super(SimpleCNN, self).__init__()
        
        # Define the layers of the CNN

        #LAYER 1
        # First convolutional layer with 1 input channel, 8 output channels, 3x3 kernel, and padding
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) 
        # First max-pooling layer with 2x2 kernel
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 

        #LAYER 2
        # Second convolutional layer with 8 input channels, 8 output channels, 3x3 kernel, and padding
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1) 
        # Second max-pooling layer with 2x2 kernel
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 

        # LAYER 3 --- Fully connected layers == lineal layers  -- HAVING 2 FULLY CONNECTED LAYERS IS NO NESESARY, BUT ITS JUST FOR EXTRACTING FEATURES
        # Fully connected layer with input size 8*7*7 and output size n_features
        self.fc1 = nn.Linear(8*7*7, n_features) 
 
        #LAYER 4
        # Fully connected layer with input size n_features and output size 1
        self.fc2 = nn.Linear(n_features, 1) 
    
    # Define the forward pass of the model
    def forward(self, x):
        # Apply the two first convolutional layers with max-pooling and ReLU activation functions
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # Reshape the data for the fully connected layers
        x = x.view(-1, 8*7*7)
        # Pass the data through the first fully connected layer (and extract features)
        extracted_features = self.fc1(x)
        # Pass the extracted features through the second fully connected layer to get the final output
        out = self.fc2(extracted_features)
        # return output and extracted features
        return out, extracted_features    

class MyCNN(object):
    
    # Constructor for the custom CNN model
    def __init__(self, n_features=8, n_epochs=25, batch_size=20, learning_rate=0.0005):
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    # Method to train the custom CNN model
    def fit(self, images, y, data_dir):
        
        # Train and validation data split
        split_ratio = 0.75
        split_index = int(len(images) * split_ratio)
        images_train = images[:split_index]
        y_train = y[:split_index]
        images_val= images[split_index:]
        y_val = y[split_index:]

        # Datasets
        train_dataset = CustomDataset(images_train, data_dir, y_train)
        val_dataset = CustomDataset(images_val, data_dir, y_val)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Create an instance of the SimpleCNN model
        self.model = SimpleCNN(n_features=self.n_features)

        # Define loss function and optimizer
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.n_epochs): #epoch--> 1 epoch is when the entire dataset is passed forward and backward through the neural network only once

            # Set the model in training mode
            self.model.train()
            # Initialize running loss
            running_loss = 0
            # Iterate over batches of training data
            for i, data in enumerate(train_loader):
                inputs, labels = data
                
                # Forward pass: Calculate model predictions and compute the loss
                outputs, _ = self.model(inputs)
                loss = criterion(outputs.squeeze(),labels.float())
                
                # Backpropagation: Zero the gradients, calculate gradients, and update the model's parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Keep track of the running loss for this epoch
                with torch.no_grad():
                    running_loss += loss.item()
            
            # Calculate the average training loss for this epoch
            train_loss = np.sqrt(running_loss/(i+1))

            # Model evaluation on the validation set
            # Set the model in evaluation mode
            self.model.eval()
            # Initialize running loss
            running_loss = 0
            # Iterate over batches of validation data
            for i, data in enumerate(val_loader):
                inputs, labels = data
                
                # Forward pass: Calculate model predictions (no gradient calculation)
                with torch.no_grad():
                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())
                    running_loss += loss.item()
            
            # Calculate the average validation loss for this epoch
            val_loss = np.sqrt(running_loss/(i+1))
            
            # Print the training and validation loss every 5 epochs
            if (epoch+1)%5 == 0:
                print("Epoch: {epoch:2d} | Train loss: {train:5.3f} | Val loss: {val:5.3f}".format(epoch=epoch+1,
                                                                                                   train=train_loss,
                                                                                                   val=val_loss))
    # Method to make predictions with trained SimpleCNN model            
    def predict(self, images, data_dir):
        
        # Create a dataset from the input images and data directory
        dataset = CustomDataset(images, data_dir)
        # Create a data loader with a batch size equal to the number of input images
        loader = DataLoader(dataset, batch_size=len(images), shuffle=False)
        
        # Initialize an array to store predicted values
        y_pred = np.zeros(len(images))
        
        # Set the model to evaluation mode (to disable features like dropout if needed)
        self.model.eval()
        # Make predictions on the input images without gradient calculation
        with torch.no_grad():
            for inputs in loader:
                y_pred, _ = self.model(inputs)
        
        # Convert the predictions to a NumPy array and reshape it
        return y_pred.numpy().reshape(-1)
    
    
    # Method to extract features from input images with trained SimpleCNN model  
    def extract_features(self, images, data_dir):
        
        # Create a dataset from the input images and data directory
        dataset = CustomDataset(images, data_dir)
        # Create a data loader with a batch size equal to the number of input images
        loader = DataLoader(dataset, batch_size=len(images), shuffle=False)
        
        # Set the model to evaluation mode (to disable features like dropout)
        self.model.eval()
        # Extract features from the input images without gradient calculation
        with torch.no_grad():
            for inputs in loader:
                _, features = self.model(inputs)
        
        # Convert the extracted features to a NumPy array
        return features.numpy()