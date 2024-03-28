import os


def check_filenames_match(images_dir, labels_dir):
    # List all files in each directory
    image_files = os.listdir(images_dir)
    label_files = os.listdir(labels_dir)

    # Extract filenames without the specific extension
    image_filenames = set([file[:-5] for file in image_files if file.endswith(".jpeg")])
    label_filenames = set([file[:-4] for file in label_files if file.endswith(".txt")])

    # Check if the sets of filenames match
    match = image_filenames == label_filenames

    # Difference (if needed for debugging or info)
    missing_in_images = label_filenames - image_filenames
    missing_in_labels = image_filenames - label_filenames

    return match, missing_in_images, missing_in_labels


# Checks if dir have same images and labels filename
_types = ["train", "validation", "test"]

for _type in _types:
    labels_dir = f"/home/ubuntu/datasets/{_type}/labels"
    images_dir = f"/home/ubuntu/datasets/{_type}/images"
    match, missing_in_images, missing_in_labels = check_filenames_match(
        images_dir, labels_dir
    )
    print(f"Filenames match for {_type} dir is {match}")
    if not match:
        print(f"Missing in images: {missing_in_images}")
        print(f"Missing in labels: {missing_in_labels}")
