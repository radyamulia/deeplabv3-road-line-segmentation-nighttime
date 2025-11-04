import os

def rename_files_sequentially(base_path, split, subfolders_to_process):
    """
    Renames files sequentially (e.g., train_1, train_2) for a given data split.
    It ensures that corresponding files (image, annotation, mask) get the same number,
    even if mask filenames have extra suffixes like '_colored' or '_visual'.
    """
    print(f"--- Processing Split: '{split}' ---")

    # The 'images' folder is our source of truth for creating the name mapping.
    source_of_truth_folder = 'images'
    source_dir = os.path.join(base_path, split, source_of_truth_folder)

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found. Skipping this split.\n")
        return

    # STEP 1: Create a mapping from old filenames to new sequential names.
    # This part remains unchanged.
    name_map = {}
    counter = 1
    for filename in sorted(os.listdir(source_dir)):
        old_base_name = os.path.splitext(filename)[0]
        new_base_name = f"{split}_{counter}"
        name_map[old_base_name] = new_base_name
        counter += 1
    
    print(f"Created a map of {len(name_map)} file names.")

    # STEP 2: Apply this mapping to all specified folders.
    for subfolder in subfolders_to_process:
        target_dir = os.path.join(base_path, split, subfolder)
        
        if not os.path.exists(target_dir):
            print(f"  - Subfolder '{subfolder}' not found. Skipping.")
            continue
        
        print(f"  - Renaming files in: '{target_dir}'")
        renamed_count = 0
        for filename in os.listdir(target_dir):
            original_base, extension = os.path.splitext(filename)
            
            # --- MODIFIED SECTION ---
            # Clean the base name by removing potential suffixes to find the correct key.
            lookup_base = original_base
            if lookup_base.endswith('_colored'):
                lookup_base = lookup_base.removesuffix('_colored')
            elif lookup_base.endswith('_visual'):
                lookup_base = lookup_base.removesuffix('_visual')
            
            # Now, look up the CLEANED name in our map
            new_base = name_map.get(lookup_base)
            # --- END MODIFIED SECTION ---
            
            if new_base:
                new_filename = f"{new_base}{extension}"
                old_filepath = os.path.join(target_dir, filename)
                new_filepath = os.path.join(target_dir, new_filename)
                
                os.rename(old_filepath, new_filepath)
                renamed_count += 1
        
        print(f"    Renamed {renamed_count} files in '{subfolder}'.")
    print(f"Finished processing split: '{split}'.\n")


# --- CONFIGURATION ---
# The name of your main dataset folder
base_path = '../datasets/v1-300-tt.voc'

# The dataset splits you want to process
# splits = ['train', 'valid', 'test']
splits = ['train', 'test']

# The subfolders within each split that need renaming.
# You can add 'masks', 'masks_overlay', etc., to this list.
subfolders_to_process = ['images', 'annotations', 'masks', 'masks_colored', 'masks_overlay']
# subfolders_to_process = ['images', 'masks']


# --- SCRIPT EXECUTION ---
for split in splits:
    rename_files_sequentially(base_path, split, subfolders_to_process)

print("✅ All specified folders have been processed.")