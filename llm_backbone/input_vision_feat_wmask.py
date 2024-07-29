"""
2024.7.17
Rewrite the embedding function in LLM
Mask some of vision tokens
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, set_seed, MistralModel, PhiModel
from transformers import TrainerCallback
from transformers import Phi3Config, Phi3ForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import MistralConfig, MistralForCausalLM
from transformers import PreTrainedModel
from .codebook import Codebook

class TLAEmbeddingMask(nn.Module):
    def __init__(self, text_embed: nn.Embedding, tokenizer: LlamaTokenizer, va_embed: Codebook, v_mask_ratio):
        super(TLAEmbeddingMask, self).__init__()

        self.text_embed = text_embed
        self.tokenizer = tokenizer
        self.va_embed = va_embed.requires_grad_(False)
        # Projector: can have other choices
        self.va_projector = nn.Linear(va_embed.embedding_dim, text_embed.embedding_dim)

        # self.v_mask_ratio = v_mask_ratio
        # self.mask_token = nn.Parameter(torch.zeros(1, text_embed.embedding_dim))

    def convert_va_ids_to_embeds(self, input_ids, id_b, id_e, device):
        """
        convert tokenizer ids of vision & action tokens to codebook embeddings.
        input_ids: Tensor(L)
        (id_b, id_e): int, int, range of ids will be converted
        """
        with torch.no_grad():
            # locate special tokens of begin and end
            p_b = torch.nonzero(torch.eq(input_ids, id_b)).item()
            p_e = torch.nonzero(torch.eq(input_ids, id_e)).item()
            ids = input_ids[p_b+1 : p_e].tolist()
            # convert tokenizer ids to codebook tokens
            ids = self.tokenizer.convert_ids_to_tokens(ids)
            # restore codebook ids from from tokens <va*>
            ids = torch.tensor([int(x[3:-1]) for x in ids], device=device) # (l)

            embeddings = self.va_embed(ids) # (l, va_embedding_dim)
        embeddings = self.va_projector(embeddings) # (l, text_embedding_dim)
        return embeddings, p_b, p_e

    # def random_masking(self, x, mask_ratio):
    #     """
    #     x: (L, D)
    #     """
    #     L, D = x.shape
    #     len_keep = int(L * (1 - mask_ratio))

    #     noise = torch.rand(L, device=x.device)

    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise) # ascend: small is keep, large is remove
    #     ids_keep = ids_shuffle[:len_keep]
        
    #     # generate bianry mask, 1 is keep
    #     mask = torch.zeros(L, device=x.device)
    #     mask[ids_keep] = 1
    #     mask = mask.unsqueeze(-1)

    #     mask_tokens = self.mask_token.repeat(L, 1)
    #     x_masked = x * mask + mask_tokens * (1 - mask)
    #     return x_masked

    def forward(self, input_ids):
        """
        input_ids: Tensor (B, L)
        output: Tensor (B, L, text_embedding_dim)
        """
        device = input_ids.device
        B, _ = input_ids.shape
        input_embeddings = self.text_embed(input_ids) # first embed token ids as originally doing
        # transform special tokens to ids
        interest_token_list = ['<bov_i>', '<eov_i>', '<boa_i>', '<eoa_i>']
        id_bov_i, id_eov_i, id_boa_i, id_eoa_i = self.tokenizer.convert_tokens_to_ids(interest_token_list)
        # map vision & action input tokens to codebook embeddings
        for b in range(B):
            cur_input_ids = input_ids[b]
            vi_embeddings, p_bov_i, p_eov_i = self.convert_va_ids_to_embeds(cur_input_ids, id_bov_i, id_eov_i, device)
            ai_embeddings, p_boa_i, p_eoa_i = self.convert_va_ids_to_embeds(cur_input_ids, id_boa_i, id_eoa_i, device)

            # # add mask to vision embeddings
            # if self.v_mask_ratio > 1e-6:
            #     vi_embeddings = self.random_masking(vi_embeddings, self.v_mask_ratio)

            # replace vision & action input embeddings
            input_embeddings[b, p_bov_i+1 : p_eov_i, :] = vi_embeddings[:, :]
            input_embeddings[b, p_boa_i+1 : p_eoa_i, :] = ai_embeddings[:, :]

        return input_embeddings


class Phi3InVisionActionFeatMask(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, tokenizer, va_embed, v_mask_ratio, **kwargs):
        # Call the parent class's from_pretrained method
        model = super(Phi3InVisionActionFeatMask, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128) # pad to multiple of 128 to improve performance
        # rewrite self.model.embed_tokens with tokenizer and vision-action model
        origin_embed_tokens = model.get_input_embeddings()
        new_embed_tokens = TLAEmbeddingMask(origin_embed_tokens, tokenizer, va_embed, v_mask_ratio)
        model.set_input_embeddings(new_embed_tokens)
        return model
    

class MistralInVisionActionFeatMask(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, tokenizer, va_embed, v_mask_ratio, **kwargs):
        # Call the parent class's from_pretrained method
        model = super(MistralInVisionActionFeatMask, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128) # pad to multiple of 128 to improve performance
        # rewrite self.model.embed_tokens with tokenizer and vision-action model
        origin_embed_tokens = model.get_input_embeddings()
        new_embed_tokens = TLAEmbeddingMask(origin_embed_tokens, tokenizer, va_embed, v_mask_ratio)
        model.set_input_embeddings(new_embed_tokens)
        return model
    