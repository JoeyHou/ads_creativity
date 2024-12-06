import gc 
import math 
import pandas as pd
import json
import re
import pickle
from collections import Counter 
            
from tqdm import tqdm 
from openai import OpenAI

from PIL import Image
import torch
# import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration

from sklearn.metrics import accuracy_score
from scipy import stats

import os
import logging
import sys
sys.path.insert(1, './src')
from models import LLaVA_VLM, T5
from env import (
    data_dir,
    mturk_data_dir
)
from prompts import (
    PROMPT_CREATIVITY,
    PROMPT_CREATIVITY_DISAGREEMENT,
    PROMPT_ATYPICALITY_OD,
    PROMPT_ATYPICALITY_DISAGREEMENT_OD,
    PROMPT_ATYPICALITY,
    PROMPT_ATYPICALITY_DISAGREEMENT,
    PROMPT_ORIGINALITY,
    PROMPT_ORIGINALITY_DISAGREEMENT,
    PROMPT_CREATIVITY_PAIRWISE,
    PROMPT_ATYPICALITY_PAIRWISE,
    PROMPT_ORIGINALITY_PAIRWISE,
)
from utils import (
    # encode_image,
    create_payload,
    query_openai,
    load_json,
    # resize_image,
    merge_images,
    get_current_timestamp
)

available_models = [
    # "liuhaotian/llava-v1.6-vicuna-13b",
    # "liuhaotian/llava-v1.5-13b",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/llava-v1.6-vicuna-13b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "gpt4",
    "Vision-CAIR/MiniGPT-4",
]

class Driver():


    ########## BEGING: INITIALIZATION ##########
    def __init__(self, config, lazy = False):
        self.config = config 
        self.debug = False if 'debug' not in config else config['debug']
        print("=> config:", config)
        self.logger = config['logger']
        self.logger.info("=> config:", config)
        self.timestamp = config['timestamp']
        
        if not lazy:
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
            elif 'gpt' in self.model_path:            
                api_key = load_json('/ihome/xli/joh227/developer/openai_key.json')['group']
                os.environ['OPENAI_API_KEY'] = api_key
                self.vlm = OpenAI()
                self.api_key = api_key
            else:
                raise NotImplementedError
            # self.parsing_model = T5(self.quantization_config)
        
        self.output_dir = config['output_dir']
        self.use_original_atypicality = False if 'use_original_atypicality' not in config else config['use_original_atypicality']
        self.pairwise_threshold = config['pairwise_threshold']
        if self.use_original_atypicality:
            self.task_list = ["atypicality"]
            self.pairwise_threshold = 0.5
        else:
            self.task_list = [
                "originality", 
                "creativity", "atypicality"
            ] if 'task_list' not in config else config['task_list']
        
        self.logger.info('[INFO] Done initialization!')
        self.load_data()
        self.logger.info('Done loading data!')
        return
        
    def load_data(self):
        if self.use_original_atypicality:
            data_path = "./data/"
            self.intrinsic_data = pd.read_csv(data_path + "modeling_instrinsic_atypicality_train.csv")
            self.pairwise_data = {
                k: pd.read_csv(data_path + "modeling_{}_diff_train.csv".format(k))
                for k in self.task_list
            }
        else:
            data_path = "./data/mturk_data/subset_0.5/" if 'data_path' not in self.config else self.config["data_path"]
            self.intrinsic_data = pd.read_csv(data_path + "modeling_instrinsic_data.csv")
            self.pairwise_data = {
                k: pd.read_csv(data_path + "modeling_{}_average_diff.csv".format(k))
                for k in self.task_list
            }
    ########## END: INITIALIZATION ##########
    

    ########## BEGING: PREDICTION ##########
    def vlm_completion(self, image_paths, prompt, temperature = 0.75):
        if len(image_paths) == 2: # image pair
            merged_image = merge_images(image_paths)
            merged_image.save('./cache/tmp_merged_pairwise.jpg')
            image_paths = ['./cache/tmp_merged_pairwise.jpg']
            
        if 'gpt' in self.model_path:
            payload = create_payload(image_paths, prompt, temperature = temperature)
            response = query_openai(payload, self.api_key)
            try:
                vlm_output = response['choices'][0]['message']['content']
            except:
                vlm_output = '[GENERATION FAILED]'
        elif 'llava' in self.model_path:
            vlm_output = self.vlm.generate(prompt, image_paths[0], self.config, temperature)
        else:
            raise NotImplementedError
        return vlm_output.replace(prompt.strip(), '')
        
    def post_processing(self, vlm_output, task):
        parsed_output = vlm_output.lower()
        match = re.search(r'answer:{0,1} {0,1}\d{1}', parsed_output)
        if match is not None:
            return int(match.group(0).strip()[-1])
        else:
            return '[PARSING FAILED]\n' + parsed_output

    def predict_instrinsic_single(self, ads_id, task):
        self.logger.warning('ads_id: ' + ads_id + '; task: ' + task)
        # sample_size = 1 if 'sample_size' not in self.config else self.config['sample_size']
        if self.use_original_atypicality:
            sample_size = 3
        elif self.debug:
            sample_size = 5
        else:
            sample_size = 25
        max_attempt = 5 if 'max_attempt' not in self.config else self.config['max_attempt']
        
        # 0. load and prep data 
        if self.use_original_atypicality:
            image_paths = [data_dir + 'original_data/' + ads_id]
        else:
            image_paths = [mturk_data_dir + 'subset_0.5/' + ads_id]
        
        if task == 'creativity':
            majority_prompt = PROMPT_CREATIVITY
            disagreement_prompt = PROMPT_CREATIVITY_DISAGREEMENT
        elif task == 'atypicality':
            if self.use_original_atypicality:
                majority_prompt = PROMPT_ATYPICALITY_OD
                disagreement_prompt = PROMPT_ATYPICALITY_DISAGREEMENT_OD
            else:
                majority_prompt = PROMPT_ATYPICALITY
                disagreement_prompt = PROMPT_ATYPICALITY_DISAGREEMENT
        elif task == 'originality':
            majority_prompt = PROMPT_ORIGINALITY
            disagreement_prompt = PROMPT_ORIGINALITY_DISAGREEMENT
        else:
            raise NotImplementedError
            
        # 1. predict single label
        counter_attempts = 0
        pred_labels = []
        ## 1.1 fix-temerature prediction 
        while True:
            vlm_output_label = self.vlm_completion(
                image_paths, 
                majority_prompt, 
                temperature = self.config['temperature']['single']
            )
            pred_label = self.post_processing(vlm_output_label, task)
            counter_attempts += 1
            if "I'm sorry" in vlm_output_label and counter_attempts <= max_attempt:
                continue 
            elif isinstance(pred_label, str) and counter_attempts <= max_attempt:
                # if self.debug: 
                self.logger.warning('[PARSING FAILED] try again')
                continue 
            else:
                break
        # if self.debug:
        self.logger.warning('=> single label prediction (fixed-temperature): ' + str(pred_label))
        self.logger.warning('=> vlm_output: ' + str(vlm_output_label))
        self.logger.warning('=> total attempts fix-temerature prediction ({}): {}'.format(task, counter_attempts))
        self.logger.warning('\n\n')

        ## 1.2 distribution prediction 
        counter_attempts = 0
        pred_distribution = []
        if 'gpt' not in self.model_path:
            while len(pred_distribution) < sample_size:
                vlm_output_label = self.vlm_completion(
                    image_paths, 
                    majority_prompt, 
                    temperature = self.config['temperature']['distribution']
                )
                pred_label = self.post_processing(vlm_output_label, task)
                counter_attempts += 1
                if "I'm sorry" not in vlm_output_label and isinstance(pred_label, int): 
                    pred_distribution.append((pred_label, vlm_output_label))
                elif counter_attempts <= max_attempt * sample_size:
                    continue 
                else:
                    break 
            # if self.debug: 
            self.logger.warning('distribution prediction')
            for pred_label, vlm_output_label in pred_distribution:
                self.logger.warning('=> label prediction: ' + str(pred_label))
                self.logger.warning('=> vlm_output: ' + str(vlm_output_label))
                # self.logger.warning()
            self.logger.warning('=> total attempts distribution prediction  ({}): {}'.format(task, counter_attempts))
            self.logger.warning('\n\n')
            
        # 2. predict disagreement
        counter_attempts = 0
        while True:
            vlm_output_disagreement = self.vlm_completion(
                image_paths, 
                disagreement_prompt, 
                temperature = self.config['temperature']['single']
            )
            pred_disagreement = self.post_processing(vlm_output_disagreement, 'disagreement')
            counter_attempts += 1
            if "I'm sorry" in vlm_output_disagreement and counter_attempts <= max_attempt:
                continue 
            elif isinstance(pred_disagreement, str) and counter_attempts <= max_attempt:
                self.logger.warning('[PARSING FAILED] try again')
                continue 
            else:
                break
        # if self.debug: 
        self.logger.warning('=> pred_disagreement: ' + str(pred_disagreement))
        self.logger.warning('=> vlm_output_disagreement: ' + str(vlm_output_disagreement))
        self.logger.warning('=> total attempts on disagreement ({}): {}'.format(task, counter_attempts))
        self.logger.warning('\n\n')

        return {
            'labels': (pred_label, vlm_output_label),
            'label_distribution': pred_distribution,
            'disagreements': (pred_disagreement, vlm_output_disagreement)
        }

    def predict_pairwise_single(self, ads_id_1, ads_id_2, task):
        # print('\n\n------------------------------------------')
        self.logger.warning('ads_id: ' + ads_id_1 + ', ' + ads_id_2  + '; task: ' + task)
        max_attempt = 5 if 'max_attempt' not in self.config else self.config['max_attempt']
        
        image_pair = [data_dir + 'original_data/' + ads_id_1, data_dir + 'original_data/' + ads_id_2] # TODO: merge two images
        
        if 'creativity' in task:
            pairwise_prompt = PROMPT_CREATIVITY_PAIRWISE
        elif 'atypicality' in task:
            pairwise_prompt = PROMPT_ATYPICALITY_PAIRWISE
        elif 'originality' in task:
            pairwise_prompt = PROMPT_ORIGINALITY_PAIRWISE
        else:
            raise NotImplementedError
        counter_attempts = 0
        while True:
            vlm_output_pairwise = self.vlm_completion(
                image_pair, 
                pairwise_prompt, 
                temperature = self.config['temperature']['single']
            )
            pred_label = self.post_processing(vlm_output_pairwise, task + '_pairwise')
            counter_attempts += 1
            if "I'm sorry" in vlm_output_pairwise and counter_attempts <= max_attempt:
                continue 
            elif isinstance(pred_label, str) and counter_attempts <= max_attempt:
                self.logger.warning('[PARSING FAILED] try again')
                continue 
            else:
                break
        
        return {
            'ads_ids': (ads_id_1, ads_id_2),
            'labels': (pred_label, vlm_output_pairwise)
        }

    
    def predict_batch(self, instrinsic = True, pairwise = True):
        debug_size = 0 if 'debug_size' not in self.config else config['debug_size']
        batch_pred = {
            'intrinsic': None,
            'pairwise': None
        }
        ## Instrinsic ## 
        if instrinsic:
            counter = 0
            # vlm_predictions = []
            instrinsic_pred = {task: [] for task in self.task_list}
            for i in tqdm(range(self.intrinsic_data.shape[0])):
                # if i == 0: continue
                ads_id = self.intrinsic_data.iloc[i]['ads_id']
                self.logger.warning('========================= [DEBUG] ads_id: ' + ads_id + ' =========================')
                for task in self.task_list:
                    vlm_pred = self.predict_instrinsic_single(ads_id, task)
                    vlm_pred['true_average'] = self.intrinsic_data.iloc[i][task + '_average']
                    true_distribution = self.intrinsic_data.iloc[i][task + '_distribution']
                    if isinstance(true_distribution, list):
                        vlm_pred['true_distribution'] = true_distribution[:len(vlm_pred['label_distribution'])]
                    elif isinstance(true_distribution, str):
                        vlm_pred['true_distribution'] = [int(i) for i in eval(true_distribution)[:len(vlm_pred['label_distribution'])]]
                    vlm_pred['true_disagreement'] = self.intrinsic_data.iloc[i][task + '_disagreement']
                    instrinsic_pred[task].append(vlm_pred)
                    
                    ## update checkpoint
                    batch_pred['intrinsic'] = instrinsic_pred
                    if self.config['save_results']:
                        pickle.dump(batch_pred, open(self.output_dir + 'pickles/batch_pred_{}_{}.pkl'.format(self.model_path.replace('/', '-'), self.timestamp), 'wb'))
                counter += 1
                if self.debug and counter >= debug_size: break
            for task in self.task_list:
                counts = Counter([dp['labels'][0] for dp in batch_pred['intrinsic'][task]])
                print(counts)
            exit(0)

        ## Pairwise 
        self.logger.info('================== Pairwise Task ==================')
        if pairwise:
            pairwise_pred = {task: [] for task in self.task_list}
            for task in self.task_list:
                counter = 0
                skip_counter = 0
                if self.use_original_atypicality: 
                    self.pairwise_data[task].sample(frac=1).reset_index(drop=True)

                # first, get the list of ad id pairs
                ads_id_pairs = []
                for i in range(self.pairwise_data[task].shape[0]):
                    if 'existing_ads_ids' in self.config and self.config['existing_ads_ids'] != '':
                        existing_data = pickle.load(open(self.config['existing_ads_ids'], 'rb'))
                        existing_ads_ids = [pred['ads_ids'] for pred in existing_data['pairwise']['atypicality']]
                    else:
                        existing_ads_ids = []
                    
                    ads_id_1, ads_id_2 = self.pairwise_data[task]['ads_pair'].values[i].split(', ')
                    if tuple([ads_id_1, ads_id_2]) in existing_ads_ids or tuple([ads_id_2, ads_id_1]) in existing_ads_ids: 
                        skip_counter += 1
                        # counter += 1
                        continue 
                    target = self.pairwise_data[task][task + '_average_diff'].values[i]
                    if abs(target) < self.pairwise_threshold: 
                        continue 
                        
                    ads_id_pairs.append((ads_id_1, ads_id_2, i))

                self.logger.warning('===== Pairwise task: {}, count: {}, skipped: {} ====='.format(task, len(ads_id_pairs), skip_counter))
                for (ads_id_1, ads_id_2, i) in tqdm(ads_id_pairs):
                    target = self.pairwise_data[task][task + '_average_diff'].values[i]
                    vlm_pred = self.predict_pairwise_single(ads_id_1 = ads_id_1, ads_id_2 = ads_id_2, task = task + '_pairwise')
                    vlm_pred['true'] = target
                    target_label = 1 if target > 0 else 2
                    self.logger.warning('=> true: ' + str(target_label))
                    self.logger.warning('=> pred: ' + str(vlm_pred['labels'][0]))
                    self.logger.warning('=> vlm_output: ' + str(vlm_pred['labels'][1]))
                    # self.logger.warning('=> skip_counter: ' + str(skip_counter))
                    self.logger.warning('\n\n')
                    pairwise_pred[task].append(vlm_pred)

                    ## update checkpoint
                    batch_pred['pairwise'] = pairwise_pred
                    if self.config['save_results']:
                        pickle.dump(batch_pred, open(self.output_dir + 'pickles/batch_pred_{}_{}.pkl'.format(self.model_path.replace('/', '-'), self.timestamp), 'wb'))
                    
                    counter += 1
                    if self.debug and counter > debug_size: break 
                    if self.use_original_atypicality and counter >= self.config['use_original_atypicality_limit']: break
                
        self.batch_pred = batch_pred
        
        return batch_pred
    ########## END: PREDICTION ##########

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    if len(sys.argv) > 1:
        config_fp = sys.argv[1]
    else:
        config_fp = 'config/config.json'
        
    with open(config_fp, 'r') as f:
        config = json.load(f)
    timestamp = get_current_timestamp()
    logging.basicConfig(
        filename="log/{}_{}.log".format(timestamp, config['model_path'].replace('/', '-')),
        format='%(asctime)s %(message)s',
        filemode='w'
    )
    logger = logging.getLogger()
    if config['debug']: 
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
        
    logger.warning(str(config))
    config['logger'] = logger
    config['timestamp'] = timestamp
    
    driver = Driver(config, lazy = False)
    driver.load_data()
    # print('done loading data!')

    run_intrinsic = True if 'intrinsic' not in config else config['intrinsic']
    run_pairwise = True if 'pairwise' not in config else config['pairwise']

    batch_pred = driver.predict_batch(
        instrinsic = run_intrinsic, 
        pairwise = run_pairwise
    )


    