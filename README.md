# Requirements
Install all the requirements by running the following command:
```pip install -r requirements.txt```
Note that the deepspeed-mii package (version 0.2.3) is not available for phi-3 but available for mistral. Besides, also note that the tokenizer in deepspeed-mii does not support outputting the special tokens added to the tokenizer, and we can solve this issue by modifying the 
/anaconda3/envs/vla/lib/python3.10/site-packages/mii/modeling/tokenizers.py: 50 to ```len(self.tokenizer)```

# Usage
Run ```deepspeed --num_gpus 1 scripts/predict_tats_tokenized_mii.py yaml/config_rt1.yaml```
You should modify the config file to specify the model path, data path, and other parameters:
- model_name_or_path: path to the LLM model
- weight_path: path to the VQ model
- save_dir: where to save the evaluation results
- src_filepath: path to the data for evaluation, it should be a jsonl file, each line with a json object with the following keys:
  - "dataset": str, the dataset name
  - "trajectory_id": int, the id of the trajectory
  - "view": int, the id of the view
  - "num_frames": int, the number of frames in the view
  - "task_description": str, the task description
  - "descriptions": a dictionary describing the frame differences
  - "image_indices": a list of indices of the frames in the view, typically being [0, 1, 2, ..., num_frames-1], however, the length of the video clips may not be divided by 6, so we need to pad the video clips to the length of 6, thus the final several frames are repeated.
  - "actions": a list of actions, each being a vector with 7 elements
  - "mean", "std": mean and std of the actions in the specified dataset

# Update by Sun Shaofan
Use ```pipeline_ssf.py```
- generate_img_paths() is for debugging, it should be rewritten or the data should be pre-processed to cover the "image_paths" field.
- the update of code mainly lies in the initialization of model and in line 83 to 92 of call_vla()  
- an example of config file can refer to ./testcodes/config_test_ssf.yaml

VQ model
- refer to ./tokenizer/tats_dinov2_action.py
- please pay attention to ./tokenizer/modules/dinov2_vit.py. 
  - revise the value of DINOv2_MODEL_PATH to your Dino V2 path. You can download the weights from:
    - [huggingface](https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m/tree/main)
  - also note that the weight is NOT used. I just copy the initialization code for training. If possible, you can revise the code to avoid loading the original weights of Dino V2.

LLM
- refer to ./llm_backbone/input_vision_feat_wmask.py

# Update by Sun Shaofan 10.8
NEW 4-layer Action Decoder
- in ./tokenizer/tats_dinov2_action.py line 732.