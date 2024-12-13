import json
from tqdm import tqdm
from pyhealth.medcode import InnerMap

icd9cm: InnerMap = InnerMap.load("ICD9CM")
icd10cm: InnerMap = InnerMap.load("ICD10CM")

def lookup_icd_code(code: str):    
    # look up ICD CM code
    try:
        value = icd9cm.lookup(code)
    except Exception:
        try:
            value = icd10cm.lookup(code)
        except Exception:
            value = None
    return value

def generate_code_id_map(dataset_sample, id_to_icd_map_path, icd_to_id_map_path, code_to_icd_map_path):
    icd_set = set()
    code_to_icd_map = {}
    for patient in tqdm(dataset_sample, desc="code map"):
        conditions = patient['conditions']
        for condition in conditions:
            for code in condition:
                value = lookup_icd_code(code)
                if value is None:
                    continue

                icd_set.add(value)
                code_to_icd_map[code] = value

    # sort icd set
    icd_list = list(icd_set)
    icd_list.sort()

    # create icd map
    icd_map = {}
    icd_map_reverse = {}
    for i in range(len(icd_list)):
        icd_map[icd_list[i]] = i
        icd_map_reverse[i] = icd_list[i]

    # save data to json
    with open(id_to_icd_map_path, "w") as f:
        json.dump(icd_map_reverse, f, indent=4)

    with open(icd_to_id_map_path, "w") as f:
        json.dump(icd_map, f, indent=4)

    # sort ICD CODE map
    code_to_icd_map = dict(sorted(code_to_icd_map.items(), key=lambda x: x[0]))

    with open(code_to_icd_map_path, "w") as f:
        json.dump(code_to_icd_map, f, indent=4)

    print("icd dict length: ", len(icd_map))