import os
from pyhealth.utils import load_pickle

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocess.data_config import DataConfig
from data_preprocess.code_id_map import generate_code_id_map
from data_preprocess.diagnoses import process_patient_diagnoses_data
from data_preprocess.ehr_data import generate_ehr_data
from data_preprocess.ehr_data import generate_trunced_ehr_data
from data_preprocess.node_masks import generate_node_masks_dict
from data_preprocess.cfipf_encode import generate_cfipf_feature
from data_preprocess.cfipf_encode import generate_node_embeddings

def load_dataset_from_pickle(input_path):
    mimic_sample = load_pickle(input_path)
    return mimic_sample

def preprocess_data(config: DataConfig):
    dataset_path = config.dataset_path
    output_path = config.output_path
    patient_dataset_path = config.patient_dataset_path
    id_to_icd_map_path = config.id_to_icd_map_path
    icd_to_id_map_path = config.icd_to_id_map_path
    code_to_icd_map_path = config.code_to_icd_map_path
    ehr_data_path = config.ehr_data_path
    trunced_ehr_data_path = config.trunced_ehr_data_path
    node_masks_path = config.node_masks_path
    cfipf_dict_path = config.cfipf_dict_path
    cfipf_matrix_path = config.cfipf_matrix_path
    print("dataset:", config.dataset)

    dataset_sample = None

    # create dictionary if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(patient_dataset_path):
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        process_patient_diagnoses_data(dataset_sample, patient_dataset_path)

    if (not os.path.exists(id_to_icd_map_path)) or (not os.path.exists(icd_to_id_map_path)) or (not os.path.exists(code_to_icd_map_path)):
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        generate_code_id_map(dataset_sample, id_to_icd_map_path, icd_to_id_map_path, code_to_icd_map_path)

    if not os.path.exists(ehr_data_path):
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        generate_ehr_data(dataset_sample, ehr_data_path)
        generate_trunced_ehr_data(dataset_sample, trunced_ehr_data_path, remain_visit_num=5)

    if not os.path.exists(node_masks_path):
        generate_node_masks_dict(patient_dataset_path, icd_to_id_map_path, node_masks_path)

    if not os.path.exists(cfipf_dict_path):    
        if dataset_sample is None:
            dataset_sample = load_dataset_from_pickle(dataset_path)
        generate_cfipf_feature(dataset_sample, node_masks_path, id_to_icd_map_path, code_to_icd_map_path, cfipf_dict_path)

    if not os.path.exists(cfipf_matrix_path):
        generate_node_embeddings(id_to_icd_map_path, cfipf_matrix_path)

if __name__ == "__main__":
    # for MIMIC-III
    config = DataConfig("MIMIC3")
    preprocess_data(config)

    # for MIMIC-IV
    config = DataConfig("MIMIC4")
    preprocess_data(config)
