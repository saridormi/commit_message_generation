import pytorch_lightning as pl
from lr_logger_callback import LearningRateLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model.encoder_decoder_module import EncoderDecoderModule
from dataset_utils.cmg_data_module import CMGDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(**cfg.dataset)
    dm.setup()

    encoder_decoder = EncoderDecoderModule(**cfg.model, tokenizer=dm._tokenizer, num_epochs=cfg.trainer.max_epochs,
                                           num_batches=len(dm.train_dataloader()) + 2*100)

    # freeze codebert encoder
    for param in encoder_decoder.model.encoder.parameters():
        param.requires_grad = False

    # freeze encoder and decoder
    for param in encoder_decoder.model.parameters():
        param.requires_grad = False

    # unfreeze cross-attention layers
    for layer in encoder_decoder.model.decoder.roberta.encoder.layer:
        for param in layer.crossattention.parameters():
            param.requires_grad = True
    # unfreeze lm head
    for param in encoder_decoder.model.decoder.lm_head.parameters():
        param.requires_grad = True

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger.watch(encoder_decoder, log='gradients', log_freq=250)

    lr_logger = LearningRateLogger()

    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger, callbacks=[lr_logger])
    # -----------------------
    #       tune lr         -
    # -----------------------
    trainer.tune(model=encoder_decoder, datamodule=dm)
    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(encoder_decoder, dm)
    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    main()