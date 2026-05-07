from pathlib import Path
import random
import shutil

def create_train_test_split(input_path, split_ratio=0.9, seed=42):
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return

    if not input_path.is_dir():
        print(f"Error: Not a directory: {input_path}")
        return

    output_path = input_path / "AUV_Datasets"
    train_path = output_path / "train"
    test_path = output_path / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Collect all JPG files
    jpg_files = [
        file_path for file_path in input_path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in [".jpg", ".jpeg"]
        and "AUV_Datasets" not in file_path.parts
    ]

    if not jpg_files:
        print("No JPG/JPEG images found.")
        return

    random.seed(seed)
    random.shuffle(jpg_files)

    split_index = int(len(jpg_files) * split_ratio)
    train_files = jpg_files[:split_index]
    test_files = jpg_files[split_index:]

    print("\n===== JPG IMAGE STATS =====")
    print(f"Input path       : {input_path}")
    print(f"Total images     : {len(jpg_files)}")
    print(f"Train images     : {len(train_files)}")
    print(f"Test images      : {len(test_files)}")
    print(f"Output folder    : {output_path}")

    def copy_flat(files, destination_root):
        for idx, file_path in enumerate(files):
            # Handle duplicate filenames by adding index
            new_name = f"{file_path.stem}_{idx}{file_path.suffix}"
            destination_file = destination_root / new_name

            shutil.copy2(file_path, destination_file)

    copy_flat(train_files, train_path)
    copy_flat(test_files, test_path)

    print("\nDataset split completed successfully.")
    print(f"Train folder: {train_path}")
    print(f"Test folder : {test_path}")


if __name__ == "__main__":
    input_folder = input("Enter input folder path: ").strip()
    create_train_test_split(input_folder)