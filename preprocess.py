import os
from data_utils import create_patches

# ==========================================
# CONFIGURATION: Update these paths!
# ==========================================

# 1. Path to the folder where you unzipped all your .tif files
INPUT_FOLDER = "D:/ssl_training_data/all_training_data"

# 2. Path where you want to save the small .npy patches
# (The script will create this folder for you)
OUTPUT_FOLDER = "./train_patches"

# 3. Size of the squares (64 is best for your current setup)
PATCH_SIZE = 64

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    print("🚀 Starting Preprocessing...")

    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: The folder '{INPUT_FOLDER}' was not found.")
    else:
        # Run the patching logic from data_utils.py
        create_patches(
            input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER, patch_size=PATCH_SIZE
        )

        print("\n✨ Done! You are now ready to run 'python train.py'")
