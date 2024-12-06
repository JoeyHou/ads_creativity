from vllm import LLM
import PIL

def test_multi_image(img_1_path, img_2_path):
    # Refer to the HuggingFace repo for the correct format to use
    prompt = "<|user|>\n<image>\n<image>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"
    # prompt = "<|user|>\n<image_1>\n<image_2>\nWhich image is more creative?<|end|>\n<|assistant|>\n"
    
    # Load the images using PIL.Image
    image1 = PIL.Image.open(img_1_path)
    image2 = PIL.Image.open(img_2_path)
    
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": [image1, image2]
        },
    })
    
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def test_batched_image(llm, image_lst):
    
    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    batch_input = [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": PIL.Image.open(img_path)},
        } for img_path in image_lst
    ]
    outputs = llm.generate(batch_input)
    
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == '__main__':
    img_1_path = './data/mturk_data/subset_0.5/0/109120.jpg'
    img_2_path = './data/mturk_data/subset_0.5/0/139270.jpg'
    
    model_path = "llava-hf/llava-v1.6-vicuna-13b-hf"
    ###### test 1: batched input ######
    llm = LLM(
        model=model_path, 
        limit_mm_per_prompt={"image": 2},
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code = True,
        # quantization="fp8"
    )
    # print('\n\n [INFO] testing batched inference')
    # test_batched_image(llm, [img_1_path, img_2_path])
    # del llm

    ###### test 2: multi image input ######
    # llm = LLM(
    #     model="microsoft/Phi-3.5-vision-instruct",
    #     trust_remote_code=True,  # Required to load Phi-3.5-vision
    #     max_num_batched_tokens=4096,
    #     max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
    #     limit_mm_per_prompt={"image": 2},  # The maximum number to accept
    # )
    
    print(['\n\n [INFO] testing multi image inference'])
    test_multi_image(img_1_path, img_2_path)
