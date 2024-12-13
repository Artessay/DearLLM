from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

def collate_fn_dict(batch):
    data = {key: [d[key] for d in batch] for key in batch[0] if key != "graph"}
    graph = [d["graph"] for d in batch]
    graph_batch = Batch.from_data_list(graph)
    return data, graph_batch


def get_dataloader(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict,
    )
    return dataloader