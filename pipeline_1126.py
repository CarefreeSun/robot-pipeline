import json
import argparse
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import os
import argparse
from tokenizer import VQGANVisionActionEval
from llm_backbone import MistralInVisionActionFeatMask, Codebook
# from vla import get_VLA_dataset
from configs import H4ArgumentParser, DataArguments, VLAModelArguments, TATSModelArguments
from torchvision import transforms
from collections import OrderedDict
from transformers import LlamaTokenizer
from safetensors import safe_open

import os
import json

def load_safetensors_weights(model, checkpoint_dir): 
    weights_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')] 
    for weights_file in weights_files: 
        weights_path = os.path.join(checkpoint_dir, weights_file) 
        with safe_open(weights_path, framework="pt", device='cpu') as f: 
            for key in f.keys():
                if 'embed_tokens' in key:
                    if key in model.state_dict().keys():
                        print('Load key: {}, Shape: {}'.format(key, model.state_dict()[key].shape))
                        model.state_dict()[key].copy_(f.get_tensor(key))
                    else:
                        print('Skip key {}'.format(key))
    return model

def generate_img_paths():
    image_paths = []
    image_path_format = './test_images/outputimage_0_{}_0.png'
    for i in range(6):
        image_paths.append(image_path_format.format(i))
    return image_paths

@torch.no_grad()
def encode(instance_data, model, tats_args, device):
    # please implement Crop in pre-processing
    # action should be a clip of 3 frames
    # img should be 2 frames, which are the start and end of the clip
    
    transform = transforms.Compose([
        transforms.Resize((tats_args.resolution, tats_args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
    ])
    video = []
    for img_path in instance_data['image_paths']:
        img = Image.open(img_path)
        img = transform(img)
        video.append(img)
    video = torch.stack(video).permute(1,0,2,3).to(device) # [C, T, H, W]
    action = torch.tensor(instance_data['actions']).to(device) # [T, 7]

    # normalize the actions
    action = (action - torch.tensor(instance_data['mean']).to(device)) / torch.tensor(instance_data['std']).to(device)

    _, _, vq_output, vq_output_action = model(video.unsqueeze(0), action.unsqueeze(0))
    video_tokens, action_tokens = vq_output['encodings'].reshape(-1), vq_output_action['encodings'].reshape(-1) # video tokens: 256, action tokens: 9*7=63

    return video_tokens, action_tokens

@torch.no_grad()
def call_vla(instance_data: dict,
             video_tokens: torch.Tensor, action_tokens: torch.Tensor, tokenizer: LlamaTokenizer,
             vla_pipe: MistralInVisionActionFeatMask, data_args: DataArguments, device):

    video_tokens = video_tokens.cpu().numpy().tolist()
    action_tokens = action_tokens.cpu().numpy().tolist()

    input_text = '<bott_i>' + instance_data['task_description'] + '<eott_i>'
    input_text += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>' + \
                '<boa_i><va0><eoa_i>'
        
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    generate_ids = vla_pipe.generate(inputs.input_ids, max_length=1280)
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    # output = vla_pipe([input_text], max_new_tokens=1024)
    # output_text = output[0].generated_text
    
    output_action_tokens_pred = [int(x[:-1]) for x in output_text.split(' <eoa_o>')[0].split('<boa_o>')[-1].split(' <va') if x != '']
    output_action_tokens_pred = torch.tensor(output_action_tokens_pred, device=device).unsqueeze(0).reshape(1, 9, 7) # output 3 clips of 3 frames, 9 in total

    output_clip_description_pred = ''

    return output_action_tokens_pred, output_clip_description_pred

def call_models(instance_data, model_vq: VQGANVisionActionEval, tokenizer: LlamaTokenizer, vla_pipe: MistralInVisionActionFeatMask,  
                tats_args: TATSModelArguments, data_args: DataArguments, device):
    
    video_tokens, action_tokens = encode(instance_data, model_vq, tats_args, device=device)

    output_action_tokens_pred, output_clip_description_pred = call_vla(instance_data, video_tokens, action_tokens, tokenizer, vla_pipe, data_args, device)

    output_action_pred = model_vq.decode_action(output_action_tokens_pred).squeeze(0).detach().cpu() # 9, 7

    instance_data['clip_description'] = output_clip_description_pred
    instance_data['actions'] = output_action_pred.tolist()

    return instance_data

def call_robot(instance_data, robot) -> dict:
    '''
    use the predicted actions to call the robot
    should override the image_paths with the observation after movement, 
    and override the given actions with the actual ones (as the robot may not be able to follow the predicted actions exactly)
    return the new instance_data 
    '''
    pass

def main():

    parser = H4ArgumentParser((VLAModelArguments, DataArguments, TATSModelArguments))
    vla_args, data_args, tats_args = parser.parse()

    local_rank = os.getenv('LOCAL_RANK', 0)
    device = f'cuda:{local_rank}'

    assert tats_args.sequence_length == 6

    # 0. define the vq model and vla model
    model_vq = VQGANVisionActionEval(tats_args)
    state_dict = torch.load(tats_args.weight_path, map_location='cpu')['state_dict']
    result = model_vq.load_state_dict(state_dict, strict=False)
    for k in result.missing_keys:
        assert 'discriminator' in k or 'perceptual_model' in k
    model_vq = model_vq.eval().to(device)

    # va_embed
    va_embed = Codebook(tats_args.n_codes, tats_args.embedding_dim)
    # reuse the state_dict above
    new_state_dict = OrderedDict()
    for key in list(state_dict.keys()):
        if key == 'codebook.embeddings':
            new_state_dict['embeddings'] = state_dict[key]
            break
    load_info = va_embed.load_state_dict(new_state_dict, strict=True)
    print(load_info)
    va_embed.eval().to(device)
    # tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(vla_args.model_name_or_path)
    vocab_size = len(tokenizer)
    # add eos token when when calling tokenizer
    visual_action_tokens_to_add = ['<va' + str(i) + '>' for i in range(0, data_args.num_visual_action_tokens)]
    num_added_visual_action_tokens = tokenizer.add_special_tokens({'additional_special_tokens': visual_action_tokens_to_add})
    special_tokens = ['<bott_i>', '<eott_i>', # task text
                        '<bots_i>', '<eots_i>', # scene text
                        '<botp_i>', '<eotp_i>', # policy text
                        '<bov_i>', '<eov_i>', '<boa_i>', '<eoa_i>', # vision and action tokens
                        '<botp_o>', '<eotp_o>', # output policy text
                        '<bov_o>', '<eov_o>', '<boa_o>', '<eoa_o>'] # output vision and action tokens
    num_added_special_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # For SFT training, padding should be on the right (if overflow occurs)
    tokenizer.padding_side = 'left'
    
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16

    model_kwargs = dict(
        # revision=model_args.model_revision,
        use_flash_attention_2=False,
        torch_dtype=torch_dtype,
        # trust_remote_code=True,
        use_cache=False
    )

    # Initialize LLM
    llm_checkpoint_path = vla_args.model_name_or_path
    model_vla = MistralInVisionActionFeatMask.from_pretrained(llm_checkpoint_path, 
                                                        tokenizer, va_embed, 0., **model_kwargs)
    # Load weights of embed_tokens
    model_vla = load_safetensors_weights(model_vla, llm_checkpoint_path).eval().to(device)

    # 1. encode the images and actions
    # the src_filepath should contain the following fields
    # task_description, image_paths
    # actions, mean, std (for normalizing the actions)
    with open(data_args.src_filepath, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        line = lines[0]

        instance_data = json.loads(line)

    instance_data['image_paths'] = generate_img_paths()

    # call the models, override original actions and clip description with the predicted ones
    instance_data = call_models(instance_data, model_vq, tokenizer, model_vla, tats_args, data_args, device)

    print(instance_data['clip_description'])
    print(instance_data['actions'])

    # call the robot, override the image_paths and actions with the actual ones
    # instance_data = call_robot(instance_data, robot)

if __name__ == '__main__':
    main()



