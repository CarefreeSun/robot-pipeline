"""
dinov2_vit.py
"""

from .base_vision import TimmViTBackbone

# Registry =>> Supported DINOv2 Vision Backbones (from TIMM) =>> Note:: Using DINOv2 w/ Registers!
#   => Reference: https://arxiv.org/abs/2309.16588
DINOv2_VISION_BACKBONES = {"dinov2-vit-l": "vit_large_patch14_reg4_dinov2.lvd142m"}
DINOv2_MODEL_PATH = {"dinov2-vit-l": "/mnt/data-rundong/huggingface/hub/models--dinov2-vit-large-patch14-reg4-lvd142m/model.safetensors"}


class DinoV2ViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            DINOv2_VISION_BACKBONES[vision_backbone_id],
            DINOv2_MODEL_PATH[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )

