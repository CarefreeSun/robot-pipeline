import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import argparse

import numpy as np
from PIL import Image

from tokenizer import VQGANVisionActionEval
from llm_backbone import MistralInVisionActionFeatMask, Codebook
# from vla import get_VLA_dataset
from configs import H4ArgumentParser, DataArguments, VLAModelArguments, TATSModelArguments
from torchvision import transforms
from collections import OrderedDict
from transformers import LlamaTokenizer
from safetensors import safe_open


from tokenizer import get_image_action_dataloader_width, ImageActionDatasetGripperWidth


class CustomCrop:
    def __init__(self, crop_param=(200, 40, 680, 680)):
        self.x = crop_param[0]
        self.y = crop_param[1]
        self.width = crop_param[2]
        self.height = crop_param[3]

    def __call__(self, img):
        return img.crop((self.x, self.y, self.x + self.width, self.y + self.height))

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

@torch.no_grad()
def encode(instance_data, model, tats_args, device):
    # action should be a clip of 3 frames
    # img should be 2 frames, which are the start and end of the clip
    # instance_data has field 'video' 'actions' 'gt_actions'
    # video is 2 images
    # actions is a clip of 3 frames
    # gt_actions is 3 clips of 3 frames, 9 in total
    # the data has been preprocessed, normalized, stacked
    
    # video = []
    # for img_path in instance_data['image_paths']:
    #     img = Image.open(img_path)
    #     img = transform(img)
    #     video.append(img)
    # video = torch.stack(video).permute(1,0,2,3).to(device) # [C, T, H, W]
    # action = torch.tensor(instance_data['actions']).to(device) # [T, 7]

    # # normalize the actions
    # action = (action - torch.tensor(instance_data['mean']).to(device)) / torch.tensor(instance_data['std']).to(device)

    video = instance_data['video'] # [B, C, 2, H, W]
    actions = instance_data['actions'] # [B, 3, 7]
    assert video.shape[0] == 1 and actions.shape[0] == 1

    _, _, vq_output, vq_output_action = model(video, actions)
    video_tokens, action_tokens = vq_output['encodings'].reshape(-1), vq_output_action['encodings'].reshape(-1) # video tokens: 256, action tokens: 9*7=63

    return video_tokens, action_tokens

@torch.no_grad()
def call_vla(instance_data: dict,
             video_tokens: torch.Tensor, action_tokens: torch.Tensor, tokenizer: LlamaTokenizer,
             vla_pipe: MistralInVisionActionFeatMask, data_args: DataArguments, device):

    video_tokens = video_tokens.cpu().numpy().tolist()
    action_tokens = action_tokens.cpu().numpy().tolist()

    print(instance_data['task_description'])

    input_text = '<bott_i>' + instance_data['task_description'][0] + '<eott_i>'
    input_text += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>' + \
                '<boa_i><va0><eoa_i>'
        
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    generate_ids = vla_pipe.generate(inputs.input_ids, max_length=1280)
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    # output = vla_pipe([input_text], max_new_tokens=1024)
    # output_text = output[0].generated_text
    
    output_action_tokens_pred = [int(x[:-1]) for x in output_text.split(' <eoa_o>')[0].split('<boa_o>')[-1].split(' <va') if x != '']
    output_action_tokens_pred = torch.tensor(output_action_tokens_pred, device=device).unsqueeze(0).reshape(1, 9, 7) # output 3 clips of 3 frames, 9 in total

    return output_action_tokens_pred

def call_models(instance_data, model_vq: VQGANVisionActionEval, tokenizer: LlamaTokenizer, vla_pipe: MistralInVisionActionFeatMask,  
                tats_args: TATSModelArguments, data_args: DataArguments, device):
    
    gt_actions = instance_data['gt_actions'] # [1, 9, 7]
    assert gt_actions.shape[0] == 1
    
    video_tokens, action_tokens = encode(instance_data, model_vq, tats_args, device=device)

    output_action_tokens_pred = call_vla(instance_data, video_tokens, action_tokens, tokenizer, vla_pipe, data_args, device)

    output_action_pred = model_vq.decode_action(output_action_tokens_pred).detach().cpu() # 1, 9, 7
    assert output_action_pred.shape[0] == 1

    # instance_data['clip_description'] = output_clip_description_pred
    # instance_data['actions'] = output_action_pred.tolist()

    return output_action_pred, gt_actions

# 设置日志
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# 正向推理和计算损失
def validate(model_vq: VQGANVisionActionEval, tokenizer: LlamaTokenizer, vla_pipe: MistralInVisionActionFeatMask,  
                tats_args: TATSModelArguments, data_args: DataArguments, dataloader, criterion, device, logger):
    model_vq.eval()
    vla_pipe.eval()
    running_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for data in dataloader:
            cnt += 1
            data = data.to(device)

            outputs, targets = call_models(data, model_vq, tokenizer, vla_pipe, tats_args, data_args, device)
            loss = abs(outputs - targets).mean().item()
            print(f"Validation Loss: {loss:.4f}")
            logger.info(f"Validation Loss: {loss:.4f}")
            running_loss += loss
    val_loss = running_loss / cnt
    return val_loss

# 主验证流程
def main():
    parser = argparse.ArgumentParser()

    # trainer args
    parser.add_argument("--nodes", type=int, default=1, help="nodes")
    parser.add_argument("--devices", type=int, default=8, help="e.g., gpu number")
    parser.add_argument("--default_root_dir", type=str, required=True, help="default_root_dir")
    parser.add_argument("--max_steps", type=int, default=100000, help="max_steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="load checkpoint model")
    parser.add_argument("--continue_train", action="store_true", help="when specified and checkpoint exists, continue training from the checkpoint")

    # model args
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_codes', type=int, default=16384)
    parser.add_argument('--n_hiddens', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--downsample', nargs='+', type=int, default=(2, 16, 16))
    parser.add_argument('--disc_channels', type=int, default=64)
    parser.add_argument('--disc_layers', type=int, default=3)
    parser.add_argument('--discriminator_iter_start', type=int, default=5000)
    parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
    parser.add_argument('--image_gan_weight', type=float, default=0.2)
    parser.add_argument('--video_gan_weight', type=float, default=0.2)
    parser.add_argument('--l1_weight', type=float, default=4.0)
    parser.add_argument('--l1_action_weight', type=float, default=10.0)
    parser.add_argument('--gan_feat_weight', type=float, default=4.0)
    parser.add_argument('--perceptual_weight', type=float, default=4.0)
    parser.add_argument('--use_pixel_weight', action='store_true')
    parser.add_argument('--frame_diff_thresh', type=float, default=0.1)
    parser.add_argument('--high_weight', type=float, default=1.0)
    parser.add_argument('--low_weight', type=float, default=0.1)
    parser.add_argument('--i3d_feat', action='store_true')
    parser.add_argument('--restart_thres', type=float, default=1.0)
    parser.add_argument('--no_random_restart', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'])
    parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])
    parser.add_argument('--action_dim', nargs='+', type=int, default=(1,1,1,1,1,1,1), help='number of action dimention, xyz, rpy, gripper')
    parser.add_argument('--action_activation', nargs='+', type=str, default=('none', 'none', 'none', 'none', 'none', 'none', 'sigmoid'),
                            help='activation function for action, should be an attribute of torch or none')
    parser.add_argument('--action_hidden_dim', type=int, default=128, help='hidden dimention of action')
    parser.add_argument('--video_action_layers', type=int, default=12, help='number of attention layers')
    parser.add_argument('--action_mask', action='store_true', help='mask action')
    parser.add_argument('--action_mask_ratio', type=float, default=0.1, help='mask ratio for action')
    parser.add_argument('--wo_transformer_residual', action='store_true', help='use transformer residual')

    # data args
    parser.add_argument("--data_root", type=str, default="/mnt/data-rundong/robot_datasets/tokenizer-training")
    parser.add_argument("--dataset_names", nargs='+', type=str, 
                        default=("bridge2", "rt1"))
    parser.add_argument("--image_root", nargs='+', type=str, 
                        default=("/mnt/robotdata/bridge2/images_bridge",
                                "/mnt/robotdata/RT1/RT1-images"))
    parser.add_argument("--normalize", action="store_true", help="normalize the actions")
    parser.add_argument("--sequence_length", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--val_check_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_step_frequency', type=int, default=2000)
    parser.add_argument('--crop', action='store_true', help="crop the input image before resize")
    parser.add_argument('--crop_param', nargs='+', type=int, default=(200, 40, 680, 680), help='(x, y, width, height)')

    args = parser.parse_args()
    # 初始化分布式训练
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # 设置日志记录
    logger = setup_logger()

    # 创建数据集和dataloader
    test_dataset = ImageActionDatasetGripperWidth(args, split='test', action=True, return_mean_std=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, sampler=test_sampler, num_workers=4) # 8卡并行，每张卡只处理1条数据，为了简便

    # 定义模型、损失函数
    parser = H4ArgumentParser((VLAModelArguments, DataArguments, TATSModelArguments))
    vla_args, data_args, tats_args = parser.parse()

    # 0. define the vq model and vla model
    model_vq = VQGANVisionActionEval(args)
    assert (args.load_checkpoint is not None)
    print(f"Load VQ checkpoint from {args.load_checkpoint}.")
    state_dict = torch.load(args.load_checkpoint, map_location='cpu')['state_dict']
    result = model_vq.load_state_dict(state_dict, strict=False)
    print(result)
    # for k in result.missing_keys:
    #     assert 'discriminator' in k or 'perceptual_model' in k
    model_vq = model_vq.eval().to(device)

    # va_embed
    va_embed = Codebook(args.n_codes, args.embedding_dim)
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
    print(f"Load LLM checkpoint from {llm_checkpoint_path}.")
    model_vla = MistralInVisionActionFeatMask.from_pretrained(llm_checkpoint_path, 
                                                        tokenizer, va_embed, 0., **model_kwargs)
    # Load weights of embed_tokens
    model_vla = load_safetensors_weights(model_vla, llm_checkpoint_path).eval().to(device)

    criterion = nn.L1Loss()

    # 进行验证
    test_loss = validate(model_vq, tokenizer, model_vla,  
                tats_args, data_args, test_loader, criterion, device, logger)
    
    logger.info(f"Validation Loss: {test_loss:.4f}")

    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
