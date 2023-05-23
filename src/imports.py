import wget
from pathlib import Path
import os
import pytorch_lightning as pl
from datasets import Dataset, load_dataset, DatasetDict, load_metric
import torch.utils.data as td
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    BertTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
from torchinfo import summary
import torch.nn as nn
from copy import deepcopy
import evaluate
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from functools import partial