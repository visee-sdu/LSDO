import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.clip_transformer import CLIPTransformer
from modules.metrics import t2v_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer
from modules.tokenization_clip import SimpleTokenizer


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None


    tokenizer = SimpleTokenizer()

    test_data_loader = DataFactory.get_data_loader(config, split_type='test')
    model = CLIPTransformer(config)

    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, t2v_metrics, None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")
    trainer.validate()


if __name__ == '__main__':
    main()
