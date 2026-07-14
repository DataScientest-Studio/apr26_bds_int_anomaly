import joblib
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from anomalib.deploy import OpenVINOInferencer
import cv2
import numpy as np
from huggingface_hub import snapshot_download

device = None
model = None

# REPLACE path with the lokal dataset path ".../MVTec_AD/archive"
DATASET_ROOT_PATH = Path(r"C:...\MVTec_AD\archive")

# example image path // replace with image to test for
IMAGE_PATH = DATASET_ROOT_PATH / "metal_nut" / "test" / "color" / "015.png"



LABEL_MAP = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet', 4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut', 8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush', 12: 'transistor', 13: 'wood', 14: 'zipper'}
DEFECT_TYPES = {'bottle': ['broken_large', 'broken_small', 'contamination'],
                'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation'],
                'capsule': ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
                'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
                'grid':['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
                'hazelnut': ['crack', 'cut', 'hole','print'],
                'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
                'metal_nut': ['bent', 'color', 'flip', 'scratch'],
                'pill': ['color', 'combined', 'contamination', 'crack', 'faulty_imprint', 'pill_type', 'scratch'],
                'screw': ['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top'],
                'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
                'toothbrush': ['defective'],
                'transistor': ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
                'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
                'zipper': ['broken_teeth', 'combined', 'fabric_border', 'fabric_interior', 'rough', 'split_teeth', 'squeezed_teeth']}




def load_patchcore_model(category):
    """ Looks for a specific PatchCore model on the projects HuggingFace repo.
    Copies the files to a local cache or uses allready exisiting files without downloading them again.
    Creates a OpenVINO model via the model.xml file and returns the inferencer object ready to use for predictions."""

    # project and subfolder paths
    repo_id = "CodingBricks/liora_mvtec_ad_project"
    subfolder_path = f"patchcore_{category}/weights/openvino"

    # load model from HuggingFace repo
    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=f"{subfolder_path}/*")
    print(f"\n\nFile downloaded to: {local_dir}/{subfolder_path}")

    # create and return the OpenVINO inferencer object
    xml_file_path = Path(local_dir) / subfolder_path / "model.xml"
    return OpenVINOInferencer(path=xml_file_path)


def extract_extrafeatures(img_path, model, device):
    """Extract hybrid-features (combination of CLS and averaged patches).

    Detects sowohl bigger, global errors as well as fine, local defects.
    """
    # ImageNet tranformation
    transform = T.Compose([
        T.Resize((518, 518)), # multiple of 14
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # get features of the very last transformer layer (index -1)
        # n_last_blocks=1 returns list with one tensor
        layer_output = model.get_intermediate_layers(tensor, n=1, return_class_token=True)
        
        # DINOv2 just produces Patch-Features (without CLS-Token)
        # Shape of layer_output: [Batch_Size, Number_of_Patches, 1024]
        # at resolution 518x518 and Patch-Size 14x14 there are exactly 1369 Patches
        patch_tokens, cls_token = layer_output[0]

        # average over all image-patches (gets fine defects)
        patch_features = torch.mean(patch_tokens, dim=1)
        
        # 3. Concatenate both vectors for a 2048 feature vector
        hybrid_feature = torch.cat((cls_token, patch_features), dim=1)
        
    return hybrid_feature.cpu().numpy().flatten()


# preload the DINOv2 for generating feature vectors
if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model is None:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()


# Step 0: extract the DINOv2 extended feature vector for the test image
dino_image_feature = extract_extrafeatures(IMAGE_PATH, model, device)

# Step 1: load object router
router_clf = joblib.load("models/object_model/object_type_router.joblib")

if __name__ == "__main__":

    # Step 2: predict object type
    object_pred = router_clf.predict([dino_image_feature])
    object_type = LABEL_MAP[object_pred[0]]

    # Step 3: load PatchCore model for anomaly detection
    # model_path = Path(f"./results/patchcore_{object_type}/weights/openvino/model.xml")
    # predictor = OpenVINOInferencer(path=model_path)

    predictor = load_patchcore_model(object_type)

    # Step 4: load test image
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 5: predict anomaly on test image
    predictions = predictor.predict(image=image)

    # Step 6: evaluate results
    # returns global anomaly score for that image
    print(f"\n\nObject {object_type} detected.")
    print(f"Anomaly-Score: {predictions.pred_score}")
    print(f"Classification: {'ANOMALY (DEFECT)' if predictions.pred_label else 'NORMAL (GOOD)'}")

    # Step 7: stop if there is no anomaly or detect the defect type
    if predictions.pred_label:
        print("The object has a defect. Predicting defect type...")
        
        if predictions.anomaly_map is not None:
            raw_map_2d = np.squeeze(predictions.anomaly_map)
            raw_map = (raw_map_2d * 255).astype(np.uint8)
            # own OpenCV-Colormap (e.g.. JET or COLORMAP_HOT)
            custom_heatmap = cv2.applyColorMap(raw_map, cv2.COLORMAP_JET)
            
            # save image
            cv2.imwrite("./results/test_image_heatmap.png", custom_heatmap)

        # Step 8: load the defect classifier
        defect_svm = joblib.load(f"models/defect_models/dinov2_svm_{object_type}_final.joblib")
        preds = defect_svm.predict([dino_image_feature])

        print(f"Object {object_type} with defect type {DEFECT_TYPES[object_type][preds[0]]}")
    else: 
         print("The object is in good condition.")