import json
from tqdm import tqdm

from data_preprocess.code_id_map import lookup_icd_code

def process_patient_diagnoses_data(dataset_sample, output_path):
    diagnoses = {}
    for patient in tqdm(dataset_sample, desc="diagnoses"):
        patient_id = int(patient['patient_id'])

        conditions = patient['conditions']
        diseases = []
        for condition in conditions:
            for code in condition:
                # look up ICD-9 CM code
                value = lookup_icd_code(code)
                if value is None:
                    continue

                diseases.append(value)
        
        # remove duplicates
        diseases = sorted(list(set(diseases)))

        if diagnoses.get(patient_id) is None:
            diagnoses[patient_id] = diseases
        else:
            print("[warning] patient id {} already exists".format(patient_id))
            diagnoses[patient_id].extend(diseases)
            diagnoses[patient_id] = sorted(diagnoses[patient_id])

    # save data to json
    diagnoses = dict(sorted(diagnoses.items(), key=lambda x: x[0]))
    print("diagnoses:", len(diagnoses))

    with open(output_path, "w") as f:
        json.dump(diagnoses, f, indent=4)
