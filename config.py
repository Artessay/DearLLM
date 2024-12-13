BATCH_SIZE = 32
HIDDEN_CHANNELS = 1024
GRAPH_VECTOR_SIZE = 8

DATASET = "MIMIC3" # "MIMIC4"
DATASET_DICTIONARY = "data/"

# generated data
GRAPH_DATASET_PATH = DATASET_DICTIONARY + DATASET + "/graph.pkl"
TRUNCED_DATA_PATH = DATASET_DICTIONARY + DATASET + "/" + "diagnoses_trunc.json"

# LLM config
LLM_MODEL_NUM = 8
LLM_MODEL_PATH = "path-to-your-llm"

PERPLEXITY_SERVER_HOST = "localhost"
PERPLEXITY_SERVER_PORT = 11225
PERPLEXITY_TEXT_TEMPLATE = \
    "{} is related to "

from data_preprocess.data_config import DataConfig
dataConfig = DataConfig(DATASET)

PATIENT_DATASET_PATH = dataConfig.patient_dataset_path 
ICD_TO_ID_MAP_PATH = dataConfig.icd_to_id_map_path
EHR_DATA_PATH = dataConfig.ehr_data_path 

class Config:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.hidden_channels = HIDDEN_CHANNELS
        self.graph_vector_size = GRAPH_VECTOR_SIZE
        self.graph_dataset_path = GRAPH_DATASET_PATH
        # self.cfipf_matrix_path = dataConfig.node_feature
        self.node_id_mask_path = dataConfig.node_masks_path

        self.llm_model_num = LLM_MODEL_NUM
        self.llm_model_path = LLM_MODEL_PATH
        self.perplexity_server_host = PERPLEXITY_SERVER_HOST
        self.perplexity_server_port = PERPLEXITY_SERVER_PORT
        self.perplexity_text_template = PERPLEXITY_TEXT_TEMPLATE