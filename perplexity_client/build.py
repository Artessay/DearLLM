import os
import json
import torch
import pickle
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

CHECK_POINT_DIR = "checkpoint/MIMIC3/"
# CHECK_POINT_DIR = "checkpoint/MIMIC4/"
CHECK_POINT_PATH = CHECK_POINT_DIR + "graph-checkpoint.pkl"
SERVER_LOG = "server.log"
out = open(SERVER_LOG, "w")

class Perplexity():
    def __init__(
            self, 
            model_name_or_path,
            text_template, 
            ehr_data_path,
            data_path, 
            code_map_path,
            output_path):

        # load data
        with open(data_path, "r") as f:
            self.diagnoses: dict = json.load(f)

        # define parameters
        self.code_map_path = code_map_path
        self.output_path = output_path

        # load disease id map
        self.disease_id_map = self.load_disease_id_map(self.code_map_path)
        self.n_disease = len(self.disease_id_map)

        # prepare patient graph map
        if os.path.exists(CHECK_POINT_PATH):
            with open(CHECK_POINT_PATH, "rb") as f:
                self.patient_graph_map = pickle.load(f)
                self.last_patient_id = list(self.patient_graph_map)[-1] + 1
        else:
            self.patient_graph_map = {}
            self.last_patient_id = 0
        
        # format prompt
        self.text_template: str = text_template
        with open(ehr_data_path, "r") as f:
            self.ehr_data: dict = json.load(f)

        # model
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name_or_path)

    def run(self):
        # produce perplexity text
        for patient_id, diseases in tqdm(self.diagnoses.items(), desc="patient", position=0):
            patient_id = int(patient_id)
            
            # skip patient that has been calculated
            if (patient_id < self.last_patient_id):
                continue

            print(datetime.now(), "Current patient id:", patient_id, file=out)
            
            # iterate through all diseases pairs
            self.edges = []
            for disease1 in diseases:
                for disease2 in diseases:
                    if disease1 == disease2:
                        continue
                    
                    self.calculate(patient_id, disease1, disease2)
            
            self.patient_graph_map[patient_id] = self.edges
            self.save_graph(CHECK_POINT_PATH)

            if len(self.patient_graph_map) % 50 == 0:
                self.save_graph((CHECK_POINT_DIR + f"graph-{patient_id}.pkl"))

        print("Consumer patient number:", len(self.patient_graph_map))
        self.save_graph(self.output_path)
        
        print("Client done")
    
    def load_disease_id_map(self, path):
        with open(path, "r") as f:
            disease_id_map = json.load(f)
        return disease_id_map
    
    def load_model_and_tokenizer(self, model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True, # False,
            trust_remote_code=True
        )
        return model, tokenizer
    
    def save_graph(self, path="graph.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.patient_graph_map, f)
    
    def calculate(self, patient_id: int, disease1: str, disease2: str):
        disease1_id = self.disease_id_map[disease1]
        disease2_id = self.disease_id_map[disease2]

        record = self.ehr_data.get(str(patient_id))
        if record is None:
            print("Error occurred when retrieving EHR data for patient: {}".format(patient_id), file=sys.stderr)
        prompt = self.text_template.format(record, disease1)

        result = self.calculate_perplexity_for_text(prompt, disease2)
        
        # print("{} [patient]: {} [result]: {} [condition1]: {} [condition2]: {}".format(datetime.now(), patient_id, result, disease1, disease2), file=out)

        self.edges.append((disease1_id, disease2_id, result))
    
    def calculate_perplexity_for_text(self, prompt: str, text: str) -> float:
        model, tokenizer = self.model, self.tokenizer
        
        # Encode the prompt and the text together
        encodings = tokenizer(prompt + text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        max_length = model.config.model_max_length
        
        # Calculate the length of the sequence after the prompt
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1) - 1 # the last token is CLS token
        prompt_len = max(prompt_len, 0)

        end_loc = min(prompt_len + max_length, seq_len)
        target_len = end_loc - prompt_len
        begin_loc = max(end_loc - max_length, 0)
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]

        target_ids = input_ids.clone()
        # Mask out the loss on the prompt
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood.to(torch.float))
        return ppl.item()

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import (
        PATIENT_DATASET_PATH, 
        ICD_TO_ID_MAP_PATH, 
        EHR_DATA_PATH,
        LLM_MODEL_PATH,
        GRAPH_DATASET_PATH, 
        PERPLEXITY_TEXT_TEMPLATE)

    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)

    client = Perplexity(
        model_name_or_path=LLM_MODEL_PATH,
        text_template=PERPLEXITY_TEXT_TEMPLATE,
        ehr_data_path=EHR_DATA_PATH,
        data_path=PATIENT_DATASET_PATH, 
        code_map_path=ICD_TO_ID_MAP_PATH,
        output_path= GRAPH_DATASET_PATH)
    
    client.run()