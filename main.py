import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import ImageDataset
from classification import ImageClassifier, train
from result_viewer import plot_loss_curves,test
from timeit import default_timer as timer 

# Set the device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset directories
train_dataset_dir = 'C:/Users/obama/OneDrive/سطح المكتب/carDataSet/DataSet/train'
test_dataset_dir = 'C:/Users/obama/OneDrive/سطح المكتب/carDataSet/DataSet/test'

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Path where the model is saved

MODEL_PATH = "C:/Users/obama/OneDrive/سطح المكتب/carDataSet/trained_model/trained_model.pth"

if __name__ == '__main__':
    # Initialize the dataset
    train_dataset = ImageDataset(root_dir=train_dataset_dir, transform=transform)
    test_dataset = ImageDataset(root_dir=test_dataset_dir, transform=transform)
    
    # Set up DataLoader
    NUM_WORKERS = os.cpu_count() 
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    #Initialize model_results (to handle both training and loading scenarios)
    model_results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Check if the model exists and load it, otherwise initialize a new model
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model = ImageClassifier().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()  # Set the model to evaluation mode
    else:
        # If model is not found, initialize and train the model
        print("Training new model...")
        model = ImageClassifier().to(device)

        # Set number of epochs
        NUM_EPOCHS = 25

        # Setup loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

        # Start the timer
        start_time = timer()

        # Train model
        model_results = train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=NUM_EPOCHS
        )

        # End the timer and print out how long it took
        end_time = timer()
        print(f"Total training time: {end_time - start_time:.3f} seconds")

        # Save the trained model
        torch.save(model.state_dict(), MODEL_PATH)

    # Plot training curves
    plot_loss_curves(model_results)

    # Test and display predictions on a random sample of test images
    test(model, test_dataset, num_images_to_show=20)

