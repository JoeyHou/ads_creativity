import gc 
import math 
import pandas as pd
import json
import re
import pickle
from tqdm import tqdm 
from openai import OpenAI

from PIL import Image
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration


import os
import logging
import sys
sys.path.insert(1, './src')
from models import LLaVA_VLM, T5
from env import (
    data_dir,
    mturk_data_dir
)
from prompts import *
from utils import (
    encode_image,
    create_payload,
    query_openai
)

available_models = [
    # "liuhaotian/llava-v1.6-vicuna-13b",
    # "liuhaotian/llava-v1.5-13b",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf"
]

class Driver():


    ########## BEGING: INITIALIZATION ##########
    def __init__(self, config):
        self.config = config 
        self.debug = False if 'debug' not in config else config['debug']
        # print("=> config:", config)
        self.logger = config['logger']
        self.load_data()
        self.logger.info('Done loading data!')
        
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit = True if 'load_in_8bit' not in config else config['load_in_8bit']
        )
        self.model_path = "llava-hf/llava-1.5-7b-hf" if 'model_path' not in config else config['model_path']

        if 'llava' in self.model_path:
            print('[INFO] initializing vlm...')
            self.vlm = LLaVA_VLM(
                self.model_path, 
                self.quantization_config
            )
            print('[INFO] done vlm initialization!')
        else:            
            api_key = load_json('/ihome/xli/joh227/developer/openai_key.json')['group']
            os.environ['OPENAI_API_KEY'] = api_key
            self.vlm = OpenAI()
        self.parsing_model = T5(self.quantization_config)

        self.output_dir = 'data/outputs/' if 'output_dir' not in config else config['output_dir']
        return
        
    def load_data(self):
        # mturk_processed_df = pd.read_csv(self.config["data_path"])
        data_path = "./data/mturk_data/subset_0.5/" if 'data_path' not in self.config else self.config["data_path"]
        self.intrinsic_data = pd.read_csv(data_path + "modeling_instrinsic_data.csv")
        self.pairwise_data = {
            k: pd.read_csv(data_path + "modeling_{}_average_diff.csv".format(k))
            for k in ["originality", "creativity", "atypicality"]
        }
    ########## END: INITIALIZATION ##########
    
    # def predict_sampling(self, ads_id, task):
    #     image_path = data_dir + 'original_data/' + ads_id
    #     sample_size = 1 if 'sample_size' not in self.config else self.config['sample_size']
    #     if sample_size == 1:
    #         temperature = 0.01
    #     else:
    #         temperature = 0.75
    #     if task == 'creativity':
    #         prompt = PROMPT_CREATIVITY
    #     elif task == 'atypicality':
    #         prompt = PROMPT_ATYPICALITY
    #     else:
    #         raise NotImplementedError
        
    #     all_pred = []
    #     all_vlm_outputs = []
    #     counter = 0
    #     while len(all_pred) < sample_size:
    #         vlm_output = self.vlm.generate(prompt, image_path, self.config, temperature)
    #         prediction = self.post_processing(vlm_output, task)
    #         counter += 1
    #         if counter > sample_size * 2: break # stop when success rate is < 50%
    #         if isinstance(prediction, str): continue 
    #         all_pred.append(prediction)
    #         all_vlm_outputs.append(vlm_output)
    #     vlm_prediction = {
    #         'task': task,
    #         'ads_id': ads_id,
    #         'pred': all_pred,
    #         'vlm_output': all_vlm_outputs
    #     }
    #     return vlm_prediction
            
    # def predict_creativity(self, data):
       
    #     counter = 0
    #     vlm_predictions = []
        
    #     for i in tqdm(range(data.shape[0])):
    #         ads_id = data.iloc[i].name 
    #         vlm_pred = self.predict_sampling(ads_id, 'creativity')
    #         true = data.iloc[i]['majority_dista']
    #         vlm_pred['true'] = true
    #         # print(vlm_pred)
    #         vlm_predictions.append(vlm_pred)
    #         counter += 1
    #         # if counter >= 3: break
    #     creativity_output_df = pd.DataFrame(vlm_predictions)
    #     creativity_output_df.to_csv(self.output_dir + 'creativity_output_df.csv', index = False)
    #     # print(disagreement_output_df)
    #     pickle.dump(creativity_output_df, open(self.output_dir + 'creativity_output_df.pkl', 'w'))
    #     return creativity_output_df
    
    ########## BEGING: POST PROCESSING ##########
    def post_processing(self, vlm_output, task):
        if task == 'creativity':
            parsing_prompt = PROMPT_CREATIVITY_PARSING.format(vlm_output = vlm_output)
        else:
            raise NotImplementedError
        
        parsed_output = self.parsing_model.parse(parsing_prompt)
        try:
            return int(re.search(r'\d{1}', parsed_output).group(0))
        except:  
            print("re.search(r'\d{1}', parsed_output).group(0)", re.search(r'\d{1}', parsed_output).group(0))
            return parsed_output


    ########## END: POST PROCESSING ##########
    
    ########## BEGING: PREDICTION ##########
    def vlm_completion(self, image_paths, prompt):
        if 'gpt' in self.model_path:
            payload = create_payload(image_paths, prompt)
            response = query_openai(payload)
            vlm_output = response['choices'][0]['message']['content']
            pred = vlm_output
        elif 'llava' in self.model_path:
            sample_size = 1 if 'sample_size' not in self.config else self.config['sample_size']
            if sample_size == 1:
                temperature = 0.01
            else:
                temperature = 0.75
            vlm_output = self.vlm.generate(prompt, image_paths[0], self.config, temperature)
            # pred = self.post_processing(vlm_output, task)
            pred = vlm_output ## TODO: add parsing here
            
        else:
            raise NotImplementedError
        return pred, vlm_output

    def predict_instrinsic_single(self, ads_id, task):
        # print('ads_id:', ads_id)
        sample_size = 1 if 'sample_size' not in self.config else self.config['sample_size']
        max_attempt = 5 if 'max_attempt' not in self.config else self.config['max_attempt']
        
        # 0. load and prep data 
        image_paths = [mturk_data_dir + 'subset_0.5/' + ads_id]
        
        # 1. predict majority label
        if task == 'creativity':
            majority_prompt = PROMPT_CREATIVITY
        elif task == 'atypicality':
            majority_prompt = PROMPT_ATYPICALITY
        elif task == 'originality':
            majority_prompt = PROMPT_ORIGINALITY
        else:
            raise NotImplementedError
        
        counter_attempts = 0
        pred_labels = []
        while True:
            pred_label, vlm_output = self.vlm_completion(image_paths, majority_prompt)
            counter_attempts += 1
            if "I'm sorry" in vlm_output and counter_attempts <= max_attempt:
                continue 
            elif len(pred_labels) < sample_size:
                pred_labels.append((pred_label, vlm_output))
            else:
                break
        if self.debug:
            print('total attempts ({}): {}'.format(task, counter_attempts))
    
        # # 2. predict average label 
        # if task == 'creativity':
        #     average_prompt = PROMPT_CREATIVITY
        # else:
        #     raise NotImplementedError
        # payload = create_payload(image_paths, average_prompt)
        # response = query_openai(payload)
        # creativity_prediction_average = response['choices'][0]['message']['content']
    
        # 3. predict disagreement
        if task == 'creativity':
            disagreement_prompt = PROMPT_CREATIVITY_DISAGREEMENT
        elif task == 'atypicality':
            disagreement_prompt = PROMPT_ATYPICALITY_DISAGREEMENT
        elif task == 'originality':
            disagreement_prompt = PROMPT_ORIGINALITY_DISAGREEMENT
        else:
            raise NotImplementedError
        
        counter_attempts = 0
        pred_disagreements = []
        while True:
            pred_disagreement, vlm_output = self.vlm_completion(image_paths, majority_prompt)
            counter_attempts += 1
            if "I'm sorry" in vlm_output and counter_attempts <= max_attempt:
                continue 
            elif len(pred_disagreements) < sample_size:
                pred_disagreements.append((pred_disagreement, vlm_output))
            else:
                break
        if self.debug:
            print('total attempts on disagreement ({}): {}'.format(task, counter_attempts))
            print()
    
        # 4. predict distribution
        ## TODO
    
        return {
            # 'creativity_majority': creativity_prediction,
            'labels': pred_labels,
            'disagreements': pred_disagreements
        }
    def predict_batch(self):

        ## Instrinsic ## 
        counter = 0
        debug_size = self.intrinsic_data.shape[0] if 'debug_size' not in self.config else config['debug_size']
        vlm_predictions = []
        intrinsic_task_lst = ["creativity", "atypicality", "originality"]
        for i in tqdm(range(self.intrinsic_data.shape[0])):
            ads_id = self.intrinsic_data.iloc[i]['ads_id']
            for task in intrinsic_task_lst:
                vlm_pred = self.predict_instrinsic_single(ads_id, task)
                vlm_pred['true'] = self.intrinsic_data.iloc[i][task + '_average']
            vlm_predictions.append(vlm_pred)
            counter += 1
            if counter >= debug_size: break
        vlm_pred_df = pd.DataFrame(vlm_predictions)
        vlm_pred_df.to_csv(self.output_dir + 'vlm_pred_df.csv', index = False)
        # print(disagreement_output_df)
        pickle.dump(vlm_pred_df, open(self.output_dir + 'vlm_pred_df.pkl', 'w'))
        return vlm_pred_df

        ## Pairwise ## 
        # TODO
        
    ########## END: PREDICTION ##########

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    if len(sys.argv) > 1:
        config_fp = sys.argv[1]
    else:
        config_fp = 'config.json'
        
    with open(config_fp, 'r') as f:
        config = json.load(f)
    
    logging.basicConfig(
        filename="info.log",
        format='%(asctime)s %(message)s',
        filemode='a'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(str(config))
    config['logger'] = logger
    
    driver = Driver(config)
    driver.load_data()
    # print('done loading data!')
    driver.predict_batch()



