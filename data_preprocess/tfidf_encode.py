import os
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pickle

def generate_cfipf_feature(dataset_sample, node_masks_path, id_to_icd_map_path, code_to_icd_map_path, cfipf_dict_path):
    text_list = []
    patient_id_index_dict = {}

    for index, sample in enumerate(dataset_sample):
        sample_text = []
        conditions = sample['conditions']
        for condition in conditions:
            for code in condition:
                sample_text.append(code)
        text_list.append(" ".join(sample_text))
        patient_id_index_dict[sample['patient_id']] = index

    vectorizer = TfidfVectorizer(lowercase=False)
    tv_fit = vectorizer.fit_transform(text_list).toarray()

    # print(tv_fit)

    with open(node_masks_path, "r") as f:
        node_masks = json.load(f)
    with open(id_to_icd_map_path, "r") as f:
        id_to_icd_map = json.load(f)
    with open(code_to_icd_map_path, "r") as f:
        code_to_icd_map = json.load(f)

    print("Vocabulary_Num: ", len(vectorizer.vocabulary_))
    print("Code Num: ", len(id_to_icd_map))
    # print(vectorizer.vocabulary_.keys())

    icd_to_code_map = {value: key for key, value in code_to_icd_map.items()}

    cfipf_dict = {}
    for patient_id in node_masks:
        row = patient_id_index_dict[patient_id]
        cfipf_dict[patient_id] = {}
        for node in node_masks[patient_id]:
            icd = id_to_icd_map[str(node)]
            code = icd_to_code_map[icd]
            # try:
            col = vectorizer.vocabulary_[code]
            tf_idf_value = tv_fit[row, col]
            cfipf_dict[patient_id][node] = tf_idf_value

    with open(cfipf_dict_path,'wb') as f:
        pickle.dump(cfipf_dict, f)

#################################################################

def prepare_text_list(dict_path):
    # create id to icd map if file does not exist
    if not os.path.exists(dict_path):
        assert False

    # load diagnose id dictionary
    with open(dict_path, "r") as f:
        icd_dict: dict = json.load(f)

    # load text list
    text_list = []
    for icd in icd_dict.values():
        text_list.append(icd)

    return text_list

def encode_text(text_list):
    vectorizer = TfidfVectorizer()
    cfipf_matrix = vectorizer.fit_transform(text_list)
    return cfipf_matrix

def generate_node_embeddings(id_to_icd_map_path, cfipf_matrix_path):
    # load text list
    text_list = prepare_text_list(id_to_icd_map_path)

    # encode text
    cfipf_matrix = encode_text(text_list)
    print("cfipf matrix shape: ", cfipf_matrix.shape)

    # save cfipf matrix
    np.save(cfipf_matrix_path, cfipf_matrix.toarray())


def test():
    from data_config import CFIPF_MATRIX_PATH

    # load cfipf matrix
    print("load cfipf matrix")
    cfipf_matrix = np.load(CFIPF_MATRIX_PATH)
    print(cfipf_matrix[0:1])
    print(cfipf_matrix[4:5])
    print(cfipf_matrix[5:6])

    from sklearn.metrics.pairwise import cosine_similarity
    print(cosine_similarity(cfipf_matrix[0:1], cfipf_matrix[4:5]))
    print(cosine_similarity(cfipf_matrix[4:5], cfipf_matrix[5:6]))