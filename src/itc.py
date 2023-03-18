from lavis import BlipITM
import torch.nn as nn
from typing import *
import torch.nn.functional as F
import torch
from dataclasses import dataclass

class Temperature(nn.Module):
    def __init__(self, value: Optional[float] = None) -> None:
        super().__init__()
        self.is_const = value is not None
        self.const = value
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.const * inputs if self.is_const else self.linear(inputs.reshape(-1, 1)).reshape(-1)

@dataclass
class ClsITCBatchData:
    images: torch.Tensor # shape: (batch_size, num_labels, <# channels>, <height>, <width>)
    text: List[str] # shape: (batch_size); texts are tokenized & moved to the same device as images in the forward of model

    @property
    def batch_size(self) -> int:
        return self.images.shape[0]

    @property
    def num_labels(self) -> int:
        return self.images.shape[1]

    @property
    def num_channels(self) -> int:
        return self.images.shape[2]

    @property
    def height(self) -> int:
        return self.images.shape[3]

    @property
    def width(self) -> int:
        return self.images.shape[4]

    @property
    def images_input(self) -> torch.Tensor: # shape: (num_labels * batch_size, _, _, _)
        return self.images.reshape(self.batch_size * self.num_labels, self.num_channels, self.height, self.width)
        
    def to(self, device):
        return ClsITCBatchData(
            images = self.images.to(device),
            text = self.text,
        )
    
    def to_json_repr(
        self, predictions: torch.Tensor, # shape: (batch_size, num_labels)
    ) -> List[Dict[str, float]]:
        # TODO: implement this
        raise NotImplementedError()


def cls_itc_collate_fn(lst: List[Dict]) -> Tuple[ClsITCBatchData, torch.Tensor]:
    """
    Custom collator for ClsITC model

    Args:
        lst (List[Dict]): list of dictionaries made by Dataset instance

    Returns:
        Tuple[ClsITCBatchData, torch.Tensor]: batch data for the model & target tensor
    """
    image_sets = []
    texts = []
    labels = []
    for item in lst:
        image_sets.append(item["images"])
        texts.append(item["text"])
        labels.append(item["label"])
    return ClsITCBatchData(
        images=torch.stack(image_sets),
        text=texts,
    ), torch.tensor(labels).long()


class ClsITC(nn.Module):
    def __init__(
            self,
            blip_itm: BlipITM,
            tokenizer_settings: Optional[Dict[str, Any]] = None,
            temperature: Optional[Temperature] = None,
            apply_softmax: bool = True,
    ) -> None:
        """
        Creates ITC classification model

        Args:
            blip_itm (BlipITM): BLIP model
            tokenizer_settings (Optional[Dict[str, Any]], optional): Tokenization named to be used on each call such as padding. Defaults to None.
                When None then the behavior is the same as in original BlipITM class
            temperature (Optional[Temperature], optional): Cosine similarity temperature. Defaults to Temperature().
            apply_softmax (bool, optional): Binary flag on whether to apply softmax to the outputs of cosine-similarities. Defaults to True.
        """
        super().__init__()
        self.tokenizer: nn.Module = blip_itm.tokenizer
        if tokenizer_settings is None:
            tokenizer_settings = {
                "padding": "longest",
                "truncation": True,
                "max_length": blip_itm.max_txt_len,
                "return_tensors": "pt",
            }
        self.tokenizer_settings = tokenizer_settings
        self.text_encoder: nn.Module = blip_itm.text_encoder
        self.text_proj: nn.Module = blip_itm.text_proj
        self.visual_encoder: nn.Module = blip_itm.visual_encoder
        self.vision_proj: nn.Module = blip_itm.vision_proj
        self.temperature = temperature
        self.apply_softmax = apply_softmax

    def forward(self, batch_data: ClsITCBatchData) -> torch.Tensor:
        text = self.tokenizer(batch_data.text, **self.tokenizer_settings).to(batch_data.images.device)
        text_encoded = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_feat = F.normalize(self.text_proj(text_encoded.last_hidden_state[:, 0, :]), dim=-1) # (batch_size, hs)
        image_encoded = self.visual_encoder.forward_features(batch_data.images_input)
        image_feat = F.normalize(self.vision_proj(image_encoded[:, 0, :]), dim=-1) # (batch_size * num_labels, hs)
        image_feats_batched = image_feat.reshape(batch_data.batch_size, batch_data.num_labels, -1) # (bs, nl, hs)
        batched_sims = torch.einsum("ij,ikj->ik", text_feat, image_feats_batched) # (bs, ns)
        if self.temperature is not None:
            batched_sims = self.temperature(batched_sims.reshape(-1)).reshape(batch_data.batch_size, -1)
        if self.apply_softmax:
            batched_sims = torch.softmax(batched_sims, 1) # (bs, *ns)
        return batched_sims
