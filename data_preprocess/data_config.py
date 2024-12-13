DATASET = "MIMIC3"
DATA_DICTIONARY = "data/"

# original data
MIMIC3_DATA_PATH = DATA_DICTIONARY + "data/sample_dataset_mimiciii_mortality_multifea.pkl"
MIMIC4_DATA_PATH = DATA_DICTIONARY + "data/sample_dataset_mimiciv_mortality_multifea.pkl"

# generated data
PATIENT_DATASET_PATH = "diagnoses.json"
ID_TO_ICD_MAP_PATH = "id_to_icd.json"
ICD_TO_ID_MAP_PATH = "icd_to_id.json"
CODE_TO_ICD_MAP_PATH = "code_to_icd_map.json"
EHR_DATA_PATH = "ehr_data.json"
TRUNCED_EHR_DATA_PATH = "ehr_data_trunced.json"
GRAPH_NODE_MASK_PATH = "node_masks.json"
CFIPF_DICT_PATH = "cfipf_dict.pkl"
CFIPF_MATRIX_PATH = "cfipf_matrix.npy"

class DataConfig:
    def __init__(self, dataset = DATASET):
        self.dataset = dataset
        self.output_path = DATA_DICTIONARY + dataset

        if dataset == "MIMIC3":
            self.dataset_path = MIMIC3_DATA_PATH
        elif dataset == "MIMIC4":
            self.dataset_path = MIMIC4_DATA_PATH
        else:
            assert False and "Dataset not support"

        self.patient_dataset_path = DATA_DICTIONARY + dataset + "/" + PATIENT_DATASET_PATH
        self.id_to_icd_map_path = DATA_DICTIONARY + dataset + "/" + ID_TO_ICD_MAP_PATH
        self.icd_to_id_map_path = DATA_DICTIONARY + dataset + "/" + ICD_TO_ID_MAP_PATH
        self.code_to_icd_map_path = DATA_DICTIONARY + dataset + "/" + CODE_TO_ICD_MAP_PATH
        self.ehr_data_path = DATA_DICTIONARY + dataset + "/" + EHR_DATA_PATH
        self.trunced_ehr_data_path = DATA_DICTIONARY + dataset + "/" + TRUNCED_EHR_DATA_PATH
        self.node_masks_path = DATA_DICTIONARY + dataset + "/" + GRAPH_NODE_MASK_PATH
        self.cfipf_dict_path = DATA_DICTIONARY + dataset + "/" + CFIPF_DICT_PATH
        self.cfipf_matrix_path = DATA_DICTIONARY + dataset + "/" + CFIPF_MATRIX_PATH

        self.mimic3_dataset_path = MIMIC3_DATA_PATH
        self.mimic4_dataset_path = MIMIC4_DATA_PATH