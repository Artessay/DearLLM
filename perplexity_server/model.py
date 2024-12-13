import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

class LargeLanguageModel:
    def __init__(self, model_name_or_path, index=0):
        # model
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name_or_path)

        # index information, used for debugging
        self.index = index

    def load_model_and_tokenizer(self, model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            mirror='tuna'
        )
        model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True
        )
        return model, tokenizer
    
    def calculate_perplexity(self, text: str) -> float:
        model, tokenizer = self.model, self.tokenizer
        encodings = tokenizer(text, return_tensors="pt")

        max_length = model.config.model_max_length
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    
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

        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()
    
    def __str__(self) -> str:
        return "LargeLanguageModel(index={})".format(self.index)
    
if __name__ == '__main__':
    from utils import get_model_name_or_path    
    model_name_or_path = get_model_name_or_path()
    model = LargeLanguageModel(model_name_or_path)
    print(model)
    
    text = "The patient was admitted to the hospital for a stroke."
    ppl = model.calculate_perplexity(text)
    print(ppl)

    text_A = "Nice to "
    text_B = "see you here."
    text_C = "Good morning"
    ppl = model.calculate_perplexity_for_text(text_A, text_B)
    print(ppl)
    ppl = model.calculate_perplexity_for_text(text_A, text_C)
    print(ppl)

    # exit()
    print("################")

    ######################################

    ehr_data_path = "data/MIMIC3/ehr_data.json"

    import json
    with open(ehr_data_path, "r") as f:
        ehr_data: dict = json.load(f)

    print(len(ehr_data))

    text_template = \
        "The data format includes a sequence of hospital visits for a single patient. " \
        "Each visit lists the diagnosed diseases, and the number of days since the last visit." \
        "The sequence of visits for this patient is as follows:\n {}\n" \
        "Based on the patient's medical history, {} is related to "

    record = ehr_data[str(17)]

    # case 1
    condition1 = "Personal history of venous thrombosis and embolism"
    condition2 = "Long-term (current) use of anticoagulants"

    prompt = text_template.format(record, condition1)
    # print(prompt, condition2)
    # print(condition1, condition2)

    ppl = model.calculate_perplexity_for_text(prompt, condition2)
    print(ppl)
    ppl = model.calculate_perplexity(condition1 + " is related to " + condition2)
    print(ppl)

    # case 2
    condition1 = "Ostium secundum type atrial septal defect"
    condition2 = "Personal history of other diseases of circulatory system"

    prompt = text_template.format(record, condition1)
    # print(prompt, condition2)
    # print(condition1, condition2)

    ppl = model.calculate_perplexity_for_text(prompt, condition2)
    print(ppl)
    ppl = model.calculate_perplexity(condition1 + " is related to " + condition2)
    print(ppl)

    record = ehr_data[str(109)]

    # case 3
    condition1 = "Personal history of venous thrombosis and embolism"
    condition2 = "Long-term (current) use of anticoagulants"

    prompt = text_template.format(record, condition1)
    # print(prompt, condition2)
    # print(condition1, condition2)

    ppl = model.calculate_perplexity_for_text(prompt, condition2)
    print(ppl)
    ppl = model.calculate_perplexity(condition1 + " is related to " + condition2)
    print(ppl)

    # case 4
    condition1 = "Ostium secundum type atrial septal defect"
    condition2 = "Personal history of other diseases of circulatory system"

    prompt = text_template.format(record, condition1)
    # print(prompt, condition2)
    # print(condition1, condition2)

    ppl = model.calculate_perplexity_for_text(prompt, condition2)
    print(ppl)
    ppl = model.calculate_perplexity(condition1 + " is related to " + condition2)
    print(ppl)