import os
import torch
import random

def to_rgb(image):
    return torch.cat([image, image, image], dim=0)

def apply_background_color(image_rgb, color):
    """Replace all black (0) background pixels with the given RGB color tuple."""
    
    # Background mask: only where all channels are zero
    background_mask = (image_rgb == 0).all(dim=0)

    for c in range(3):
        image_rgb[c][background_mask] = color[c]
    return image_rgb

def process_dataset(dataset, alpha_odd=0.01, alpha_even=0.02):
    
    new_data = []

    odd_indices = [i for i, (_, label) in enumerate(dataset) if label % 2 == 1]
    even_indices = [i for i, (_, label) in enumerate(dataset) if label % 2 == 0]

    num_odd_to_color = int(alpha_odd * len(odd_indices))
    num_even_to_color = int(alpha_even * len(even_indices))

    selected_odd = set(random.sample(odd_indices, num_odd_to_color))
    selected_even = set(random.sample(even_indices, num_even_to_color))

    for i, (img, label) in enumerate(dataset):
        rgb_img = to_rgb(img)
        binary_label = torch.tensor(1 if label % 2 == 1 else 0)

        if i in selected_odd:
            rgb_img = apply_background_color(rgb_img, color=(0.0, 0.0, 1.0))  # Blue
        
        elif i in selected_even:
            rgb_img = apply_background_color(rgb_img, color=(1.0, 0.0, 0.0))  # Red

        new_data.append((rgb_img, binary_label))

    return new_data

def main(data_dir="data", alpha_odd=0.01, alpha_even=0.02):
    
    train_data = torch.load(os.path.join(data_dir, "mnist_train.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "mnist_test.pt"), weights_only=False)

    processed_train = process_dataset(train_data, alpha_odd, alpha_even)
    processed_test = process_dataset(test_data, alpha_odd, alpha_even)

    torch.save(processed_train, os.path.join(data_dir, "mnist_train_colored.pt"))
    torch.save(processed_test, os.path.join(data_dir, "mnist_test_colored.pt"))

if __name__ == "__main__":
    main()