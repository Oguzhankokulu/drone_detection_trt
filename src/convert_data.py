import os
import pathlib
from PIL import Image
from tqdm import tqdm
import shutil


BASE_DIR = pathlib.Path("data/visdrone_raw")
OUTPUT_DIR = pathlib.Path("data/visdrone_yolo")
SPLIT_MAP = {
    "train" : "VisDrone2019-DET-train",
    "val" : "VisDrone2019-DET-val",
    "test" : "VisDrone2019-DET-test-dev"
}

# Mapping: 0:Person, 1:Vehicle, 2:Cycle
CLASS_MAPPING = {
    0: 0, 1: 0,
    3: 1, 4: 1, 5: 1, 8: 1,
    2: 2, 6: 2, 7: 2, 9: 2
}

def convert_box(size, box):
    """
    Converts VisDrone box to YOLO normalized format.
    
    Args:
        size: (img_width, img_height)
        box: (x_min, y_min, w_box, h_box)
    
    Returns:
        (x_center, y_center, w_norm, h_norm)
    """
    # Unpack the inputs
    img_width, img_height = size
    x_min, y_min, w_box, h_box = box

    # Calculate centers
    x_center = x_min + w_box/2
    y_center = y_min + h_box/2

    # Normalize the center and size
    x_center /= img_width
    y_center /= img_height
    w_norm = w_box / img_width
    h_norm = h_box/ img_height

    return (x_center, y_center, w_norm, h_norm)

def process_split(split_name):
    """
    Process the splits (annotations and images)
    
    :param split_name: split name for the dataset (train, val, test)
    """
    image_output_path = OUTPUT_DIR / "images" / split_name
    label_output_path = OUTPUT_DIR / "labels" / split_name

    image_output_path.mkdir(parents=True, exist_ok=True)
    label_output_path.mkdir(parents=True, exist_ok=True)

    source_folder = SPLIT_MAP[split_name]
    anno_dir = BASE_DIR / source_folder / "annotations"
    src_img_dir = BASE_DIR / source_folder / "images"

    files = list(anno_dir.glob("*.txt"))
    
    for anno_file in tqdm(files):
        img_filename = anno_file.stem + ".jpg"
        src_img_path = src_img_dir / img_filename
        dst_img_path = image_output_path / img_filename
        
        if not src_img_path.exists():
            continue

        with Image.open(src_img_path) as img:
            width, height = img.size
        
        new_lines = []
        with open(anno_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split(",")

                if parts[4] == "0":
                    continue
                
                raw_class_id = int(parts[5])
                if raw_class_id not in CLASS_MAPPING:
                    continue

                new_class_id = CLASS_MAPPING[raw_class_id]

                x_min = int(parts[0])
                y_min = int(parts[1])
                w_box = int(parts[2])
                h_box = int(parts[3])

                yolo_box = convert_box((width, height), (x_min, y_min, w_box, h_box))

                new_line = f"{new_class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                new_lines.append(new_line)
        
        if new_lines:
            dst_label_path = label_output_path / anno_file.name
            with open(dst_label_path, 'w') as out_f:
                out_f.writelines(new_lines)
            
            if not dst_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    process_split("test")