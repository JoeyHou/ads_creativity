import torch
import transformers

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
    
class LLaVA_VLM:
    def __init__(
        self,
        model_path,
        quantization_config,
        offload_folder='.'
        ):
        if "v1.6" in model_path:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map='auto'
            )
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map='auto'
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
        self.model_path = model_path

    def generate(self, prompt, image_path, config = None, temperature = 0.01):
        if 'v1.6' in self.model_path:
            system_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n".format(prompt = prompt)
        else:
            system_prompt = "USER: <image>\n{prompt} ASSISTANT:".format(prompt = prompt)
        # temperature = 0 if 'temperature' not in config or config is None else config['temperature']
        do_sample = True
        
        image = Image.open(image_path)
        inputs = self.processor(system_prompt, image, return_tensors="pt").to('cuda')
        # inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Generate
        generate_ids = self.model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        result = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        # print('output:', result)
        return result

    def get_description(self, image_path):
        caption_prompt = "Describe the image advertisement in detail."
        return self.completion(caption_prompt, image_path) 

class T5():
    def __init__(self, quantization_config):
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-xl", 
            quantization_config=quantization_config,
            device_map='auto'
        )
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        
    def parse(self, input_text):
        inputs = self.t5_tokenizer(input_text, return_tensors="pt").to('cuda')
        outputs = self.t5.generate(**inputs)
        return self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        