import copy
import torch
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from pyhealth.datasets.sample_dataset import SampleEHRDataset

def dataset_merge(graph_dataset_path, patient_dataset_path, cfipf_dict_dataset_path, patient_dataset_path_merged):
    with open(graph_dataset_path, "rb") as f:
        graph_dict = pickle.load(f)
    with open(patient_dataset_path, "rb") as f:
        datasample = pickle.load(f)
    with open(cfipf_dict_dataset_path, "rb") as f:
        cfipf_dict = pickle.load(f)

    datasample_ori = copy.deepcopy(datasample)
    
    datagraph_sample = []
    filtered_ids = []
    for sample in tqdm(datasample):
        patient_id: str = sample['patient_id']
        patient_id = patient_id.replace("+", "")
        patient_id = patient_id.replace("-", "")
        
        try:
            patient_graph:list = graph_dict[int(patient_id)]
            patient_node_mask_cfipf:dict = cfipf_dict[patient_id]
        except KeyError:
            filtered_ids.append(int(patient_id))
            continue
        
        if len(patient_node_mask_cfipf) == 0 :
            filtered_ids.append(int(patient_id))
            continue

        node_mask = list(patient_node_mask_cfipf.keys())
        node_mask_dict = {node_id: i for i, node_id in enumerate(node_mask)}
        node_feature = [[node_id, patient_node_mask_cfipf[node_id]] for _, node_id in enumerate(node_mask)]

        edge_index = []
        edge_weight = []
        for edge in patient_graph:
            (start, end, weight) = edge
            
            start = node_mask_dict[start]
            end = node_mask_dict[end]
            edge_index.append([start, end])
            edge_weight.append(weight)

        edge_index = torch.tensor(edge_index, dtype=torch.int).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        x = torch.tensor(node_feature, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        sample["graph"] = data
        datagraph_sample.append(sample)

    print("All patient num: ", len(datasample_ori))
    print("Used patient num: ", len(datagraph_sample))
    print("Filtered patients: ", filtered_ids)

    with open(patient_dataset_path_merged, "wb") as f:
        pickle.dump((datasample_ori, datagraph_sample), f)
    
    return datagraph_sample

if __name__ == "__main__":
    dataset = "MIMIC3"

    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_preprocess.data_config import DataConfig
    config = DataConfig(dataset)

    graph_temp_dataset_path = "data/" + dataset + "/graph_norm.pkl"
    patient_temp_dataset_path = config.dataset_path
    cfipf_dict_temp_dataset_path = config.cfipf_dict_path
    patient_dataset_path_merged = "data/" + dataset + "/samples-graph-merged.pkl"
    sample_dataset_path = f'data/{dataset}/sample_dataset_{dataset}.pkl'

    if not os.path.exists(patient_dataset_path_merged):
        datagraph_sample = dataset_merge(graph_temp_dataset_path, patient_temp_dataset_path, cfipf_dict_temp_dataset_path, patient_dataset_path_merged)
    else:
        with open(patient_dataset_path_merged, 'rb') as f:
            (_, datagraph_sample) = pickle.load(f)
    
    with open(sample_dataset_path, 'wb') as f:
        pickle.dump(datagraph_sample, f)