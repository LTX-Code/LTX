from dataclasses import dataclass

from torch import Tensor
from transformers.modeling_outputs import SequenceClassifierOutput

from main.seg_classification.output_dataclasses.lossloss_output import LossLossOutput


@dataclass
class ImageClassificationWithTokenClassificationModelOutput:
    lossloss_output: LossLossOutput
    vit_masked_output: SequenceClassifierOutput
    masked_image: Tensor
    interpolated_mask: Tensor
    tokens_mask: Tensor
