from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from ts.torch_handler.base_handler import BaseHandler
import torch

class Falcon7BHandler(BaseHandler):
    def initialize(self, context):
        """Load the model and tokenizer during initialization."""
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
        inputs = data[0].get("body")
        if isinstance(inputs, (bytes, bytearray)):
            inputs = inputs.decode('utf-8')

        tokenized_input = self.tokenizer(inputs, return_tensors="pt", padding=True)
        return tokenized_input

    def inference(self, model_input):
        with torch.no_grad():
            output = self.model.generate(**model_input)
        return output

    def postprocess(self, inference_output):
        decoded_output = self.tokenizer.decode(inference_output[0], skip_special_tokens=True)
        return [decoded_output]
