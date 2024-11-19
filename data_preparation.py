import os
import shutil

def organize_cars_by_brand(input_dir, output_dir):

    # Dictionary to hold brand names and their corresponding model folders
    brand_dict = {}

    # Traverse the input directory to collect all brands
    for label_folder in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label_folder)
        if os.path.isdir(label_path):
            # Get the brand name as the first word of the folder name
            brand_name = label_folder.split()[0]

            # Initialize a new list for this brand if not already done
            if brand_name not in brand_dict:
                brand_dict[brand_name] = []

            # Add the model folder to the brand's list
            brand_dict[brand_name].append(label_path)

    # Copy images from each brand's model folders to the new brand folder
    for brand_name, model_folders in brand_dict.items():
        # Create a new folder for the brand in the output directory
        brand_folder_path = os.path.join(output_dir, brand_name.replace(" ", "_"))  # Replace spaces with underscores
        if not os.path.exists(brand_folder_path):
            os.makedirs(brand_folder_path)

        # Copy all images from all model folders for this brand
        for model_folder in model_folders:
            for image_file in os.listdir(model_folder):
                image_path = os.path.join(model_folder, image_file)
                # Check if the image is a file
                if os.path.isfile(image_path):
                    # Copy the image to the brand folder
                    shutil.copy2(image_path, os.path.join(brand_folder_path, image_file))

    print(f"Car images copied and organized by brand in {output_dir}")


train_original_dataset_path = 'C:/Users/obama/OneDrive/سطح المكتب/carDataSet/Stanford-Cars-dataset-main/train'
train_dataset_dir = 'C:/Users/obama/OneDrive/سطح المكتب/carDataSet/DataSet/train'
organize_cars_by_brand(train_original_dataset_path, train_dataset_dir)

# test_original_dataset_path = 'C:/Users/obama/OneDrive/سطح المكتب/carDataSet/Stanford-Cars-dataset-main/test'
# test_dataset_dir = 'C:/Users/obama/OneDrive/سطح المكتب/carDataSet/DataSet/test'
# organize_cars_by_brand(test_original_dataset_path, test_dataset_dir)
