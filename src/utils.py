import json
import jsonlines
from datetime import datetime
from pprint import pprint
import base64
import mimetypes
import os
import requests
import sys
from PIL import Image

def load_txt_prompt(filename):
    """
    Load a prompt from local txt file
    """
    prompt = ''.join(open(filename, 'r').readlines())
    return prompt

def load_json(filename):
    """
    Load a JSON file given a filename
    If the file doesn't exist, then return an empty dictionary instead
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def write_json(data, filepath):
    assert isinstance(data, dict), '[ERROR] Expect dictionary data!'
    json_string = json.dumps(data, indent = 4)
    with open(filepath, 'w') as outfile:
        outfile.write(json_string)
    # return 0

def load_jsonl(filename):
    file_content = []
    try:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                file_content.append(obj)
            return file_content
    except FileNotFoundError:
        return []

def write_jsonl(data, filepath):
    with open(filepath, 'w') as jsonl_file:
        for line in data:
            jsonl_file.write(json.dumps(line))
            jsonl_file.write('\n')

def openai_call(prompt, client, config):
    # set parameters
    temperature = config['temperature'] if 'temperature' in config else 0.75
    max_tokens = config['max_tokens'] if 'max_tokens' in config else 250
    # stop_tokens = config['stop_tokens'] if 'stop_tokens' in config else ['###']
    frequency_penalty = config['frequency_penalty'] if 'frequency_penalty' in config else 0
    presence_penalty = config['presence_penalty'] if 'presence_penalty' in config else 0
    wait_time = config['wait_time']  if 'wait_time' in config else 0
    model = config['model'] if 'model' in config else 'text-dacinvi-003'
    return_logprobs = config['return_logprobs'] if 'return_logprobs' in config else False
    logprobs = True if return_logprobs else None
    response = client.chat.completions.create(
            model=model,
            messages = prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            # stop_tokens=stop_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs
    )
    completion = response.choices[0].message.content.strip() 
    if logprobs:
        logprobs = response.choices[0].logprobs
        return completion, logprobs
    else:
        return completion

def make_prompt(data, template):
    return template.format(**data)

def protoqa_get_answers(dp):
    return {
        'keys': [ans['answers'][0] for ans in dp['answers']['clusters'].values()],
        'counts': [ans['count'] for ans in dp['answers']['clusters'].values()],
    }

def get_current_timestamp():
    return str(datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')[4:]


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

def create_payload(
        images: list[str], 
        prompt: str, 
        model="gpt-4-vision-preview", 
        max_tokens=300, 
        detail="high",
        temperature=0.75
    ):
    """Creates the payload for the API request."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "temperature": temperature,
                    "text": prompt, # TODO: check how to add temperature to openai requests
                },
            ],
        },
    ]

    for image in images:
        base64_image = encode_image(image)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": base64_image,
                "detail": detail,
            }
        })

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }


def query_openai(payload, api_key):
    """Sends a request to the OpenAI API and prints the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def resize_image(image, base_height):
    # base_width = image
    # img = Image.open('somepic.jpg')
    h_percent = (base_height / float(image.size[1]))
    w_size = int( (float(image.size[0]) * float(h_percent)) )
    image = image.resize((w_size, base_height), Image.Resampling.LANCZOS)
    return image

def merge_images(image_paths):
    images = [
        Image.open(x) for x in image_paths
    ]
    # min_width = min([img.size[0] for img in images])
    min_height = int(min([img.size[1] for img in images]))
    # im1.resize(newsize)
    images = [resize_image(img, min_height) for img in images]
    
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im