import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def init():
    global model
    global tokenizer
    global device
    global generation_config

    model_name = "wongsodillon/feedback-generator-falcon7b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generation_config = model.generation_config
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

def run(raw_data):
    data = json.loads(raw_data)
    prompt = data.get("prompt", "")

    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return json.dumps({"generated_text": generated_text})
