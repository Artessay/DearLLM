import os
import json
import pickle
import socket
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

CHECK_POINT_DIR = "checkpoint/MIMIC3/"
CHECK_POINT_PATH = CHECK_POINT_DIR + "graph-checkpoint.pkl"

class PerplexityClient():
    def __init__(
            self, 
            data_path, 
            code_map_path,
            output_path,
            num_workers, 
            server_host, 
            server_port):

        # load data
        with open(data_path, "r") as f:
            self.diagnoses: dict = json.load(f)

        # define parameters
        self.num_workers = num_workers
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

        # prepare perplexity server
        self.server_host = server_host
        self.server_port = server_port

        # multi-threading lock
        self.lock = threading.Lock()

    def run(self):
        # produce perplexity text
        for patient_id, diseases in tqdm(self.diagnoses.items(), desc="patient", position=0):
            patient_id = int(patient_id)
            
            # skip patient that has been calculated
            if (patient_id < self.last_patient_id):
                continue

            print("Current patient id:", patient_id)
            
            # multi-threaded
            pool = ThreadPoolExecutor(max_workers=self.num_workers)
            # self.graph = torch.zeros([self.n_disease, self.n_disease])
            self.edges = []
        
            # iterate through all diseases pairs
            diseases = diseases[:10] if len(diseases) > 10 else diseases
            for disease1 in diseases:
                for disease2 in diseases:
                    if disease1 == disease2:
                        continue
                    
                    _ = pool.submit(self.calculate, patient_id, disease1, disease2)
            
            pool.shutdown(wait=True)
            
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
    
    def save_graph(self, path="graph.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.patient_graph_map, f)
    
    def calculate(self, patient_id: int, disease1: str, disease2: str):
        disease1_id = self.disease_id_map[disease1]
        disease2_id = self.disease_id_map[disease2]

        result = self.send_for_calculation(patient_id, disease1, disease2)

        self.lock.acquire()
        self.edges.append((disease1_id, disease2_id, result))
        self.lock.release()

    def send_for_calculation(self, patient_id: int, disease1: str, disease2: str) -> float:
        client_socket = socket.socket()
        client_socket.connect((self.server_host, self.server_port))

        data_to_send = (patient_id, disease1, disease2)
        serialized_data = pickle.dumps(data_to_send)
        client_socket.send(serialized_data)

        result = client_socket.recv(1024).decode()
        client_socket.close()

        try:
            result = float(result)
        except Exception as e:
            result = float("inf")

        return result

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import (
        PATIENT_DATASET_PATH, 
        ICD_TO_ID_MAP_PATH, 
        PERPLEXITY_SERVER_HOST, PERPLEXITY_SERVER_PORT, 
        LLM_MODEL_NUM, GRAPH_DATASET_PATH)

    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)

    client = PerplexityClient(
        data_path=PATIENT_DATASET_PATH, 
        code_map_path=ICD_TO_ID_MAP_PATH,
        output_path=GRAPH_DATASET_PATH,
        num_workers=LLM_MODEL_NUM,
        server_host=PERPLEXITY_SERVER_HOST,
        server_port=PERPLEXITY_SERVER_PORT)
    
    client.run()