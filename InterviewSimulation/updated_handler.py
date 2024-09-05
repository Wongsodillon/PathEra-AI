from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from ts.torch_handler.base_handler import BaseHandler
import torch

class Falcon7BHandler(BaseHandler):
    def initialize(self, context):
        self.model_name = "wongsodillon/feedback-generator-falcon7b"
        
        config = PeftConfig.from_pretrained(self.model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = PeftModel.from_pretrained(self.model, self.model_name)

        self.model.eval()  

    def preprocess(self, data):
        generation_params = data[0].get("generation_config", {})
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inputs = data[0].get("body")
        if isinstance(inputs, (bytes, bytearray)):
            inputs = inputs.decode('utf-8')

        tokenized_input = self.tokenizer(inputs, return_tensors="pt", padding=True)
        return tokenized_input.to(device)

    def inference(self, model_input, generation_params):
        with torch.no_grad():
            generation_config = GenerationConfig.from_pretrained(self.model_name)
            generation_config.update(generation_params)
            output = self.model.generate(**model_input, **generation_params)
        return output

    def postprocess(self, inference_output):
        decoded_output = self.tokenizer.decode(inference_output[0], skip_special_tokens=True)
        return [decoded_output]
