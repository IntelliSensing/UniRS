# This file is modified from https://github.com/haotian-liu/LLaVA/

import sys
sys.path.append("/home/lyj/data/VILA/")
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.data.dataset import LazySupervisedDataset
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.mm_utils import process_images

from PIL import Image
import math
import numpy as np

from torchvision.transforms import Resize
from pytorchvideo.data.encoded_video import EncodedVideo

import signal

# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()
# Set the signal handler
signal.signal(signal.SIGALRM, handler)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_model_output(model, image_processor, tokenizer, video_path, qs, args):

    num_video_frames = model.config.num_video_frames
    images, video_loading_succeed = LazySupervisedDataset._load_video(video_path, num_video_frames, args)
    image_tensor = process_images(images, image_processor, model.config)

    qs = '<image>\n' * num_video_frames + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_num = [num_video_frames]

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    input_ids = torch.unsqueeze(input_ids, 0)
    input_ids = torch.as_tensor(input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            image_num=image_num,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base, args)
    args.image_processor = image_processor

    gt_questions = json.load(open(os.path.expanduser(args.gt_file_question), "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(os.path.expanduser(args.gt_file_answers), "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    args.output_dir = os.path.expanduser(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    args.video_dir = os.path.expanduser(args.video_dir)
    print(f"Video directory: {args.video_dir}")
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    # Read cache answer file, each line is a json object
    if os.path.exists(answers_file):
        cache_ans_file = open(answers_file, "r")
        cache_ans = cache_ans_file.readlines()
        cache_ans_file.close()
    else:
        cache_ans = []

    # Get cached video ids
    cache_set = set([json.loads(line)['id'] for line in cache_ans])
        

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # List to store the output results
    output_list = [] 
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']


    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        folder_name = sample['answer']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line

            if "Activitynet_Zero_Shot_QA" in args.video_dir:
                temp_path = os.path.join(args.video_dir, folder_name, f"{id.rsplit('_', 1)[0]}{fmt}")
            else:
                temp_path = os.path.join(args.video_dir, folder_name, f"{video_name}{fmt}")
            if f"{id}" in cache_set:
                print(f"Skipping {id} because it is in the cache")
                continue
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, image_processor, tokenizer, video_path, question, args)
                sample_set['pred'] = output
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                    # Write into the answer file.
                with open(answers_file, 'a') as f:
                    f.write(json.dumps(sample_set) + "\n")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--cc_n_layers', type=int, default=0)
    parser.add_argument('--cc_head', type=int, default=8)
    parser.add_argument('--cc_dropout', type=float, default=0.)
    parser.add_argument('--vision_resolution', type=int, default=384)
    args = parser.parse_args()

    eval_model(args)
