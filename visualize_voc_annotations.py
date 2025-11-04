import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

# --- Helper Functions ---

def xml_polygon_to_mask(xml_path, image_size, class_mapping, drawing_order):
    """
    Parses XML and generates a single-channel mask from polygons,
    respecting a defined drawing order.
    """
    if not os.path.exists(xml_path):
        print(f"DEBUG: XML file not found: {xml_path}")
        return None

    width, height = image_size
    mask_image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask_image)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects_by_name = {}
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in objects_by_name:
            objects_by_name.setdefault(name, [])
        objects_by_name.get(name).append(obj)

    for class_name in drawing_order:
        class_id = class_mapping.get(class_name, 0)
        
        if class_id == 0 and class_name not in class_mapping:
            print(f"WARNING: Class '{class_name}' not found in mapping for file {os.path.basename(xml_path)}. Defaulting to ID 0.")
            continue

        if class_name in objects_by_name:
            for obj in objects_by_name.get(class_name):
                polygon_element = obj.find('polygon')
                if polygon_element is not None:
                    points = []
                    for i in range(1, 100):
                        x_coord = polygon_element.find(f'x{i}')
                        y_coord = polygon_element.find(f'y{i}')
                        if x_coord is not None and y_coord is not None:
                            points.append((float(x_coord.text), float(y_coord.text)))
                        else:
                            break
                    
                    if points:
                        draw.polygon(points, fill=class_id)
                else:
                    print(f"DEBUG: Object '{class_name}' in {os.path.basename(xml_path)} has no polygon data.")

    return np.array(mask_image)

def visualize_mask(mask_array, class_colors):
    """Applies a color map to a single-channel mask array for visualization."""
    height, width = mask_array.shape
    visual_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        visual_mask_condition = (mask_array == class_id)[:, :, np.newaxis]
        visual_mask = np.where(visual_mask_condition, np.array(color, dtype=np.uint8), visual_mask)
    return Image.fromarray(visual_mask)

def visualize_mask_on_image(image_path, mask_array, class_colors, alpha=0.5):
    """Overlays a colorized mask on the original image."""
    original_image = Image.open(image_path).convert('RGB')
    mask_image = visualize_mask(mask_array, class_colors)
    
    mask_image = mask_image.resize(original_image.size)
    
    blended_image = Image.blend(original_image, mask_image, alpha=alpha)
    return blended_image

# --- Processing Function ---

def process_dataset(image_dir, xml_dir, output_training_dir, output_visual_overlay_dir, output_visual_colored_dir, class_mapping, drawing_order, class_colors):
    """
    Processes a dataset split (e.g., train or valid) to generate all three mask types.
    """
    os.makedirs(output_training_dir, exist_ok=True)
    os.makedirs(output_visual_overlay_dir, exist_ok=True)
    os.makedirs(output_visual_colored_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            base_name = os.path.splitext(xml_file)[0]
            image_file = base_name + '.jpg'
            image_path = os.path.join(image_dir, image_file)
            xml_path = os.path.join(xml_dir, xml_file)

            if not os.path.exists(image_path):
                print(f"Image not found for {xml_file}, skipping.")
                continue

            with Image.open(image_path) as img:
                image_size = img.size
            
            mask_array = xml_polygon_to_mask(xml_path, image_size, class_mapping, drawing_order)
            
            if mask_array is not None:
                # 1. Save the single-channel mask (for training)
                mask_image_for_training = Image.fromarray(mask_array, mode='L')
                mask_image_for_training.save(os.path.join(output_training_dir, base_name + '.png'))
                
                # 2. Save the colorized mask (overlay)
                visual_overlay_mask = visualize_mask_on_image(image_path, mask_array, class_colors, alpha=0.5)
                visual_overlay_mask.save(os.path.join(output_visual_overlay_dir, base_name + '_visual.png'))
                
                # 3. Save the colored-only mask
                visual_colored_mask = visualize_mask(mask_array, class_colors)
                visual_colored_mask.save(os.path.join(output_visual_colored_dir, base_name + '_colored.png'))
                
                print(f"Processed {xml_file}")

# --- Main Script ---

# --- DIRECTORY SETUP ---
BASE_DATASET_FOLDER_NAME = "../datasets/v1-300-tt.voc" 
# Training set paths
train_image_dir = f'{BASE_DATASET_FOLDER_NAME}/train/images'
train_xml_dir = f'{BASE_DATASET_FOLDER_NAME}/train/annotations'
train_output_training_dir = f'{BASE_DATASET_FOLDER_NAME}/train/masks'
train_output_visual_overlay_dir = f'{BASE_DATASET_FOLDER_NAME}/train/masks_overlay'
train_output_visual_colored_dir = f'{BASE_DATASET_FOLDER_NAME}/train/masks_colored'

# Validation set paths
# valid_image_dir = f'{BASE_DATASET_FOLDER_NAME}/valid/images'
# valid_xml_dir = f'{BASE_DATASET_FOLDER_NAME}/valid/annotations'
# valid_output_training_dir = f'{BASE_DATASET_FOLDER_NAME}/valid/masks'
# valid_output_visual_overlay_dir = f'{BASE_DATASET_FOLDER_NAME}/valid/masks_overlay'
# valid_output_visual_colored_dir = f'{BASE_DATASET_FOLDER_NAME}/valid/masks_colored'

# # Test set paths
test_image_dir = f'{BASE_DATASET_FOLDER_NAME}/test/images'
test_xml_dir = f'{BASE_DATASET_FOLDER_NAME}/test/annotations'
test_output_training_dir = f'{BASE_DATASET_FOLDER_NAME}/test/masks'
test_output_visual_overlay_dir = f'{BASE_DATASET_FOLDER_NAME}/test/masks_overlay'
test_output_visual_colored_dir = f'{BASE_DATASET_FOLDER_NAME}/test/masks_colored'


# --- CONFIGURATION ---
class_mapping = {
    'road': 1,
    'lm_solid': 2,
    'lm_dashed': 3,
}

drawing_order = ['road', 'lm_solid', 'lm_dashed']

class_colors = {
    0: (0, 0, 0),         # Background (Black)
    1: (255, 0, 0),       # Road (Red)
    2: (0, 255, 0),       # lm_solid (Green)
    3: (255, 255, 0),     # lm_dashed (Yellow)
}

# --- PROCESS BOTH DATASETS ---

print("Processing TRAINING dataset...")
process_dataset(
    train_image_dir,
    train_xml_dir,
    train_output_training_dir,
    train_output_visual_overlay_dir,
    train_output_visual_colored_dir,
    class_mapping,
    drawing_order,
    class_colors
)

# print("\nProcessing VALIDATION dataset...")
# process_dataset(
#     valid_image_dir,
#     valid_xml_dir,
#     valid_output_training_dir,
#     valid_output_visual_overlay_dir,
#     valid_output_visual_colored_dir,
#     class_mapping,
#     drawing_order,
#     class_colors
# )

print("\nProcessing TESTING dataset...")
process_dataset(
    test_image_dir,
    test_xml_dir,
    test_output_training_dir,
    test_output_visual_overlay_dir,
    test_output_visual_colored_dir,
    class_mapping,
    drawing_order,
    class_colors
)

print("\nProcessing complete.")