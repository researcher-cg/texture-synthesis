import os
import cv2
import numpy as np

def create_collage_shelf(image_folder, collage_path, max_collage_width=2000, spacing=5):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print("No images found.")
        return

    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(image_folder, f))
        if img is not None:
            images.append(img)

    shelves = []  # list of rows; each row is a list of images
    current_shelf = []
    current_width = 0
    max_height_in_shelf = 0

    for img in images:
        h, w = img.shape[:2]
        if current_width + w + spacing * (len(current_shelf)) <= max_collage_width:
            current_shelf.append(img)
            current_width += w
            max_height_in_shelf = max(max_height_in_shelf, h)
        else:
            # save current shelf and start new one
            shelves.append((current_shelf, max_height_in_shelf))
            current_shelf = [img]
            current_width = w
            max_height_in_shelf = h

    # Add last shelf
    if current_shelf:
        shelves.append((current_shelf, max_height_in_shelf))

    collage_width = max(
        sum(img.shape[1] for img in shelf[0]) + spacing * (len(shelf[0]) - 1)
        for shelf in shelves
    )
    collage_height = sum(shelf[1] for shelf in shelves) + spacing * (len(shelves) - 1)

    # Create blank canvas (black)
    collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

    y_offset = 0
    for shelf_imgs, shelf_height in shelves:
        x_offset = 0
        for img in shelf_imgs:
            h, w = img.shape[:2]
            collage[y_offset:y_offset+h, x_offset:x_offset+w] = img
            x_offset += w + spacing
        y_offset += shelf_height + spacing

    cv2.imwrite(collage_path, collage)
    print(f"Collage saved at {collage_path}")

# Example usage:
# create_collage_shelf("path/to/images", "collage_shelf.jpg", max_collage_width=1920, spacing=10)

def create_tight_collage_png(image_folder, collage_path, max_width=1920, fixed_height=200, spacing=5, max_images=20):
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in valid_exts][:max_images]

    images = []
    for file in image_files:
        img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.shape[2] == 3:  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Add alpha channel
        elif img.shape[2] == 4:
            pass  # Already has alpha
        else:
            continue

        h, w = img.shape[:2]
        scale = fixed_height / h
        resized = cv2.resize(img, (int(w * scale), fixed_height))
        images.append(resized)

    # Row packing (shelf-based layout)
    shelves = []
    current_row = []
    current_width = 0

    for img in images:
        h, w = img.shape[:2]
        if current_width + w + spacing * len(current_row) <= max_width:
            current_row.append(img)
            current_width += w
        else:
            shelves.append(current_row)
            current_row = [img]
            current_width = w
    if current_row:
        shelves.append(current_row)

    collage_width = max(sum(img.shape[1] for img in row) + spacing * (len(row)-1) for row in shelves)
    collage_height = (fixed_height + spacing) * len(shelves) - spacing

    # Transparent background
    collage = np.zeros((collage_height, collage_width, 4), dtype=np.uint8)

    y_offset = 0
    for row in shelves:
        x_offset = 0
        for img in row:
            h, w = img.shape[:2]
            collage[y_offset:y_offset + h, x_offset:x_offset + w] = img
            x_offset += w + spacing
        y_offset += fixed_height + spacing

    cv2.imwrite(collage_path, collage)
    print(f"Saved transparent collage to: {collage_path}")

# Example:
#create_collage_shelf("./arxaiaellinika/TrainScratched/", "collage_arxaiaellinika_scratched_original.jpg", max_collage_width=1920, spacing=4)
create_tight_collage_png("./arxaiaellinika/Train/", "collage_arxaiaellinika_OURS.png", max_width=1600, fixed_height=200)
