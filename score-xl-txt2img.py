import json
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from flask import jsonify
from zipfile import ZipFile

import base64

from io import BytesIO

def init():
    
    #thread tunning
    torch.set_num_threads(1)

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "stable-diffusion-xl-base-0.9.zip")

    with ZipFile(model_path, 'r') as zipObj:
        # Extract all the contents of zip file in a directory
        zipObj.extractall('stable-difussion')

    model_id = "stable-difussion/stable-diffusion-xl-base-0.9"
    device = "cuda"

    global inference_pipeline

    inference_pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True,)
    inference_pipeline = inference_pipeline.to(device)
    

def run(request):
    content = json.loads(request)

    prompt = content["prompt"]
    
    # all API parameters
    strength=0.75
    if 'strength' in content.keys():
        strength = float(content['strength'])

    guidance_scale=7.5
    if 'guidanceScale' in content.keys():
        guidance_scale = float(content['guidanceScale'])

    width=512
    if 'width' in content.keys():
        width = int(content['width'])

    height=512
    if 'height' in content.keys():
        height = int(content['height'])

    num_inference_steps=50
    if 'numInferenceSteps' in content.keys():
        num_inference_steps = float(content['numInferenceSteps'])

    mediaContentType = "PNG"
    if 'contentType' in content.keys():
        mediaContentType = str(content['contentType'])

    mediaContentType = "PNG"
    if 'contentType' in content.keys():
        mediaContentType = str(content['contentType'])
    
    #prompt = "a photo of an astronaut riding a horse on mars"
    with autocast("cuda"):
        result_image = inference_pipeline(prompt=prompt, init_image=img, strength=strength, guidance_scale=guidance_scale, num_inference_steps = num_inference_steps, width=width, height=height ).images[0]

    #keeping this as is for compatibility with current paint code
    #but I don't like it, nor I think it content type matters
    buffered_result_image = BytesIO()

    if mediaContentType == "PNG" or mediaContentType is None or mediaContentType == "":
        result_image = result_image.convert('RGBA')
        result_image.save(buffered_result_image, format="PNG")
    elif mediaContentType == "JPEG":
        result_image.save(buffered_result_image, format="JPEG")
    else:
        result_image = result_image.convert('RGBA')
        result_image.save(buffered_result_image, format="PNG")

    result_image_str = base64.b64encode(buffered_result_image.getvalue()).decode('utf-8')

    return jsonify({"result": result_image_str})
