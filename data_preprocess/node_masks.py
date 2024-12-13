import json
from tqdm import tqdm

def generate_node_masks_dict(patient_dataset_path, icd_to_id_map_path, node_masks_path):
    # load patient dataset
    with open(patient_dataset_path, "r") as f:
        patient_dataset = json.load(f)
    # print("patient number:", len(patient_dataset))

    # load icd to id map
    with open(icd_to_id_map_path, "r") as f:
        icd_to_id_map = json.load(f)
    # print("icd to id map length:", len(icd_to_id_map))

    # create node id set
    node_id_mask = {}
    for patient_id, diseases in tqdm(patient_dataset.items()):
        patient_id = int(patient_id)
        node_set = set()
        for disease in diseases:
            disease_id = icd_to_id_map[disease]
            node_set.add(disease_id)
        node_id_mask[patient_id] = list(sorted(node_set))

    # save node id set
    with open(node_masks_path, "w") as f:
        json.dump(node_id_mask, f, indent=4)
    
    print("node masks:", len(node_id_mask))

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import PATIENT_DATASET_PATH, ICD_TO_ID_MAP_PATH, GRAPH_NODE_MASK_PATH
    generate_node_masks_dict(PATIENT_DATASET_PATH, ICD_TO_ID_MAP_PATH, GRAPH_NODE_MASK_PATH)