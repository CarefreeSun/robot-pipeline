"""
Rewrite the embedding function in LLM
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, set_seed, MistralModel, PhiModel
from transformers import TrainerCallback
from transformers import Phi3Config, Phi3ForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import MistralConfig, MistralForCausalLM
from transformers import PreTrainedModel
from .codebook import Codebook

class TLAEmbedding(nn.Module):
    def __init__(self, text_embed: nn.Embedding, tokenizer: LlamaTokenizer, va_embed: Codebook):
        super(TLAEmbedding, self).__init__()

        self.text_embed = text_embed
        self.tokenizer = tokenizer
        self.va_embed = va_embed.requires_grad_(False)
        # Projector: can have other choices
        self.va_projector = nn.Linear(va_embed.embedding_dim, text_embed.embedding_dim)
    
    def forward(self, input_ids):
        """
        input_ids: Tensor (B, L)
        output: Tensor (B, L, embedding_dim)
        """
        device = input_ids.device
        B, L = input_ids.shape
        input_embeddings = self.text_embed(input_ids) # first embed token ids as originally doing
        # transform special tokens to ids
        interest_token_list = ['<bov_i>', '<eov_i>', '<boa_i>', '<eoa_i>']
        id_bov_i, id_eov_i, id_boa_i, id_eoa_i = self.tokenizer.convert_tokens_to_ids(interest_token_list)
        # map vision & action input tokens to codebook embeddings
        for b in range(B):
            with torch.no_grad():
                cur_input_ids = input_ids[b]
                # locate special tokens
                p_bov_i = torch.nonzero(torch.eq(cur_input_ids, id_bov_i)).item()
                p_eov_i = torch.nonzero(torch.eq(cur_input_ids, id_eov_i)).item()
                p_boa_i = torch.nonzero(torch.eq(cur_input_ids, id_boa_i)).item()
                p_eoa_i = torch.nonzero(torch.eq(cur_input_ids, id_eoa_i)).item()
                vi_ids = cur_input_ids[p_bov_i+1 : p_eov_i].tolist()
                ai_ids = cur_input_ids[p_boa_i+1 : p_eoa_i].tolist()
                vi_ids = self.tokenizer.convert_ids_to_tokens(vi_ids)
                ai_ids = self.tokenizer.convert_ids_to_tokens(ai_ids)
                # restore codebook ids from from tokens <va*>
                vi_ids = torch.tensor([int(x[3:-1]) for x in vi_ids], device=device)
                ai_ids = torch.tensor([int(x[3:-1]) for x in ai_ids], device=device) # (l)

                vi_embeddings = self.va_embed(vi_ids)
                ai_embeddings = self.va_embed(ai_ids)
                
            vi_embeddings = self.va_projector(vi_embeddings)
            ai_embeddings = self.va_projector(ai_embeddings) # (l, embedding_dim)
            # replace vision & action input embeddings
            input_embeddings[b, p_bov_i+1 : p_eov_i, :] = vi_embeddings[:, :]
            input_embeddings[b, p_boa_i+1 : p_eoa_i, :] = ai_embeddings[:, :]

        return input_embeddings


class Phi3InVisionActionFeat(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, tokenizer, va_embed, **kwargs):
        # Call the parent class's from_pretrained method
        model = super(Phi3InVisionActionFeat, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128) # pad to multiple of 128 to improve performance
        # rewrite self.model.embed_tokens with tokenizer and vision-action model
        origin_embed_tokens = model.get_input_embeddings()
        new_embed_tokens = TLAEmbedding(origin_embed_tokens, tokenizer, va_embed)
        model.set_input_embeddings(new_embed_tokens)
        return model
    

class MistralInVisionActionFeat(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, tokenizer, va_embed, **kwargs):
        # Call the parent class's from_pretrained method
        model = super(MistralInVisionActionFeat, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128) # pad to multiple of 128 to improve performance
        # rewrite self.model.embed_tokens with tokenizer and vision-action model
        origin_embed_tokens = model.get_input_embeddings()
        new_embed_tokens = TLAEmbedding(origin_embed_tokens, tokenizer, va_embed)
        model.set_input_embeddings(new_embed_tokens)
        return model
