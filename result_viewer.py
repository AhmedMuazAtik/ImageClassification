import torch
import random
import matplotlib.pyplot as plt

def plot_loss_curves(model_results):
    results = dict(list(model_results.items()))
    
    # Get the loss and accuracy values from the results dictionary
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    
    # Calculate number of epochs
    epochs = range(len(loss))
    
    # Plot
    plt.figure(figsize=(15, 7))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show()

def test(model, test_dataset, num_images_to_show=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Select random sample of images from the test dataset
    random_indices = random.sample(range(len(test_dataset)), num_images_to_show)
    
    # Evaluate and visualize predictions
    with torch.no_grad():
        for idx in random_indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            label = torch.tensor(label).to(device)
            
            # Predict
            output = model(image)
            _, predicted = torch.max(output, 1)
            
            # Get label names
            true_label_name = test_dataset.get_label_name(label.item())
            predicted_label_name = test_dataset.get_label_name(predicted.item())
            
            # Display image with true and predicted labels
            plt.imshow(image.squeeze(0).cpu().permute(1, 2, 0))  # Convert for display
            plt.title(f"True: {true_label_name} | Predicted: {predicted_label_name}")
            plt.axis("off")
            plt.show()
