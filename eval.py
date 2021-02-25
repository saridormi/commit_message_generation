import os

import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model.encoder_decoder_module import EncoderDecoderModule
from model.gpt2lmhead_module import GPT2LMHeadModule
from dataset_utils.cmg_data_module import CMGDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(**cfg.dataset)

    if 'ckpt_path' in cfg:
        PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
        print("Checkpoint path\n", PATH, '\n')
        model = EncoderDecoderModule.load_from_checkpoint(PATH)
    else:
        model = GPT2LMHeadModule(**cfg.model, tokenizer=dm._trg_tokenizer)

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)
    # -----------------------
    #          test         -
    # -----------------------
    if 'ckpt_path' in cfg:
        trainer.test(ckpt_path=PATH, datamodule=dm, model=model)
    else:
        trainer.test(datamodule=dm, model=model)


if __name__ == '__main__':
    main()