# DearLLM: Enhancing Personalized Healthcare via Large Language Models-Deduced Feature Correlations

Welcome to the official repository for *DearLLM: Enhancing Personalized Healthcare via Large Language Models-Deduced Feature Correlations*.


**📢 News: this work has been accepted at the AAAI 2025 !**

**If you find our project interesting or helpful, we would appreciate it if you could give us a star! Your support is a tremendous encouragement to us!**

## Requirements

All dependencies are described in the file `requirements.txt`, you can install the packages required for this project via the command below.

```sh
pip install -r requirements.txt
```

## Usage

### Prepare Dataset

As we can not provide the MIMIC-III and MIMIC-IV datasets, you must acquire the data yourself from https://mimic.physionet.org/. Please reference https://github.com/sunlabuiuc/PyHealth for more details about data preparation and put the processed data into `data` folder before entering into the folllowing steps.

1. Generate needed information from `data_preprocess` folder

    ```sh
    python data_preprocess/utils.py
    ```

2. Generate relation graph from the diagnoses result of each patient by code under `perplexity-server` folder and `perplexity-client` folder.

    ```sh
    python perplexity_server/server.py
    python perplexity_client/client.py
    ```

3. Postprocess perplexity graphs followed by instructions in `data_preprocess/data_postprocess.ipynb`, which normalize perplexity score into range [0, 1].

4. Merge patient sample data, node feature and graph information into one file.

    ```sh
    python data_preprocess/dataset_merge.py
    ```

### Run Code

To run the code, just run `main_dearllm.py` in root folder.

```sh
python main_dearllm.py
```
