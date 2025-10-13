import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image

def create_dummy_data(base_dir="dummy_mri_data", num_samples_per_class=10, num_artefact_domains=2):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    image_dir = os.path.join(base_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    artefact_domains = [f"artefact_{i}" for i in range(num_artefact_domains)]
    if num_artefact_domains >= 1: artefact_domains[0] = "noise"
    if num_artefact_domains >= 2: artefact_domains[1] = "motion"

    data = []
    img_idx = 0
    for domain_idx, domain_name in enumerate(artefact_domains):
        for class_label in [0, 1, 2]:  # Good, Moderate, Bad
            for i in range(num_samples_per_class):
                img_filename = f"img_{img_idx:04d}_domain{domain_idx}_class{class_label}.png"
                img_path = os.path.join(image_dir, img_filename)
                
                dummy_img_array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
                if class_label == 0:  # "Good"
                    dummy_img_array[30:90, 30:90] = np.random.randint(150, 250, (60,60), dtype=np.uint8)
                elif class_label == 1:  # "Moderate"
                    dummy_img_array[40:80, 40:80] = np.random.randint(100, 200, (40,40), dtype=np.uint8)
                    dummy_img_array += np.random.randint(0, 50, (128,128), dtype=np.uint8)
                else:  # "Bad"
                    dummy_img_array[50:70, 50:70] = np.random.randint(50, 150, (20,20), dtype=np.uint8)
                    dummy_img_array += np.random.randint(0, 100, (128,128), dtype=np.uint8)
                
                dummy_pil_img = Image.fromarray(dummy_img_array.astype(np.uint8)).convert('L')
                dummy_pil_img.save(img_path)
                
                data.append({"image_path": img_path, "artefact_domain": domain_name, "label": class_label})
                img_idx += 1
                
    df = pd.DataFrame(data)
    csv_path = os.path.join(base_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dummy data created at {base_dir} with {len(df)} entries.")
    print(f"Dummy CSV at {csv_path}")
    return csv_path, image_dir, artefact_domains 