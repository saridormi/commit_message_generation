import pytorch_lightning as pl
import pandas as pd
import numpy as np
import wandb
import nltk
from collections import defaultdict
from typing import Optional
from copy import copy
from transformers import EncoderDecoderModel, RobertaModel, RobertaConfig, GPT2LMHeadModel, GPT2Config, \
    RobertaTokenizer, GPT2Tokenizer
from metrics import accuracy_MRR
from datasets import load_metric
nltk.download('wordnet')


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 actual_generation: bool,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 num_layers_encoder: Optional[int] = None,
                 num_layers_decoder: Optional[int] = None,
                 encoder_name_or_path: Optional[str] = None,
                 decoder_name_or_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.actual_generation = actual_generation
        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self.save_hyperparameters()

        if encoder_name_or_path is not None and decoder_name_or_path is not None:
            # use pretrained RoBERTa as encoder
            encoder = RobertaModel.from_pretrained(encoder_name_or_path)
            # resize embeddings to match vocabulary size
            encoder.resize_token_embeddings(len(self._src_tokenizer))
            # remove layers if necessary
            if num_layers_encoder is not None and num_layers_encoder < encoder.config.num_hidden_layers:
                encoder = EncoderDecoderModule.remove_layers_from_model(encoder, num_layers_encoder, is_gpt=False)

            # use pretrained GPT-2 as decoder
            config = GPT2Config.from_pretrained(decoder_name_or_path)
            config.is_decoder = True
            config.add_cross_attention = True
            decoder = GPT2LMHeadModel.from_pretrained(decoder_name_or_path, config=config)
            # remove layers if necessary
            if num_layers_decoder is not None and num_layers_decoder < decoder.config.n_layer:
                decoder = EncoderDecoderModule.remove_layers_from_model(decoder, num_layers_decoder, is_gpt=True)

        elif num_layers_decoder is not None and num_layers_encoder is not None:
            # use randomly initialized RoBERTa as encoder
            encoder_config = RobertaConfig()
            encoder_config.num_hidden_layers = num_layers_encoder
            encoder = RobertaModel(config=encoder_config)
            # resize embeddings to match vocabulary size
            encoder.resize_token_embeddings(len(self._src_tokenizer))

            # use randomly initialized GPT-2 as decoder
            decoder_config = GPT2Config()
            decoder_config.n_layer = num_layers_decoder
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            decoder = GPT2LMHeadModel(config=decoder_config)
        else:
            raise ValueError("You have to specify either num_layers for training from scratch \
                                                  or paths for loading pretrained models")

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # set decoding params
        self.model.config.no_repeat_ngram_size = 4
        self.model.config.early_stopping = True
        self.model.config.num_beams = 4

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.meteor = load_metric("meteor")

        self.table_data = defaultdict(list)

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch['msg_input_ids'])
        return self.model(input_ids=batch['diff_input_ids'],
                          attention_mask=batch['diff_attention_mask'],
                          decoder_input_ids=batch['msg_input_ids'],
                          decoder_attention_mask=batch['msg_attention_mask'],
                          labels=batch['msg_labels'])

    def generate(self, batch):
        return self.model.generate(input_ids=batch['diff_input_ids'],
                                   attention_mask=batch['diff_attention_mask'],
                                   decoder_input_ids=batch['generation_input_ids'],
                                   decoder_attention_mask=batch['generation_attention_mask'],
                                   max_length=batch['msg_input_ids'].shape[1])

    def test_step(self, batch, batch_idx):
        if self.actual_generation:
            return self.actual_generation_step(batch)
        else:
            return self.next_token_metrics_step(batch)

    def actual_generation_step(self, batch):
        gen_sequence = self.generate(batch)
        gen_input_len = batch['generation_input_ids'].shape[1]
        gen_sequence = [i[gen_input_len:] for i in gen_sequence.tolist()]  # leave only generated part

        targets_no_history = batch['msg_input_ids'].detach().clone().to(self.device)
        targets_no_history[batch['msg_labels'] == -100] = self._trg_tokenizer.pad_token_id

        decoded_source = self.decode_src(batch['diff_input_ids'])[0]
        decoded_targets_no_history, decoded_history, decoded_preds = \
            self.decode_trg(targets_no_history,
                            batch['generation_input_ids'],
                            gen_sequence)

        # add data to a little table with examples
        self.table_data["Diff"].extend(decoded_source)
        self.table_data["History (generation input)"].extend(decoded_history)
        self.table_data["Target"].extend(decoded_targets_no_history)
        self.table_data["Prediction"].extend(decoded_preds)

        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in decoded_preds],
                            references=[[line.split()] for line in decoded_targets_no_history])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)

    def next_token_metrics_step(self, batch):
        scores = self(batch).logits

        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, batch['msg_labels'], top_k=5, shift=True)

        return {"acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5}

    def test_epoch_end(self, outputs):
        if self.actual_generation:
            bleu = self.bleu.compute()
            rouge = self.rouge.compute()
            meteor = self.meteor.compute()

            df = pd.DataFrame.from_dict(self.table_data)
            table = wandb.Table(dataframe=df)

            self.logger.experiment.log({"test_examples": table,
                                        "test_bleu": bleu["bleu"],
                                        "test_rouge1": rouge["rouge1"].mid.fmeasure,
                                        "test_rouge2": rouge["rouge2"].mid.fmeasure,
                                        "test_rougeL": rouge["rougeL"].mid.fmeasure,
                                        "test_meteor": meteor["meteor"]}, step=self.examples_count)
        else:
            df = pd.DataFrame.from_dict({'acc_top1': [x["acc_top1"] for x in outputs],
                                         'acc_top5': [x["acc_top5"] for x in outputs],
                                         'MRR_top5': [x["MRR_top5"] for x in outputs]})
            wandb.Table.MAX_ROWS = 15000
            table = wandb.Table(dataframe=df)
            self.logger.experiment.log({"test_metrics_for_CI": table,
                                        "test_acc_top1": np.mean([x["acc_top1"] for x in outputs]),
                                        "test_acc_top5": np.mean([x["acc_top5"] for x in outputs]),
                                        "test_MRR_top5": np.mean([x["MRR_top5"] for x in outputs])}, step=10000001)

    def decode_src(self, *args):
        return tuple(self._src_tokenizer.batch_decode(arg, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False) for arg in args)

    def decode_trg(self, *args):
        return tuple(self._trg_tokenizer.batch_decode(arg, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False) for arg in args)

    @staticmethod
    def remove_layers_from_model(teacher, num_layers, is_gpt):
        if not is_gpt:
            teacher_config = teacher.config
            student_config = copy(teacher.config)
            student_config.num_hidden_layers = num_layers
            student = RobertaModel(config=student_config)

            # copy all embeddings
            student.embeddings.word_embeddings = teacher.embeddings.word_embeddings
            student.embeddings.position_embeddings = teacher.embeddings.position_embeddings
            student.embeddings.token_type_embeddings = teacher.embeddings.token_type_embeddings
            student.embeddings.LayerNorm = teacher.embeddings.LayerNorm
            student.embeddings.dropout = teacher.embeddings.dropout

            # uniformly pick from middle layers from teacher
            # it is basically np.linspace(0, teacher_config.num_hidden_layers,
            #                             num=student_config.num_hidden_layers, endpoint=True)
            step = (teacher_config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
            for student_layer, teacher_layer in enumerate(
                    int(i * step) for i in range(student_config.num_hidden_layers)):
                student.encoder.layer[student_layer] = teacher.encoder.layer[teacher_layer]

        else:
            teacher_config = teacher.config
            student_config = copy(teacher.config)
            student_config.n_layer = num_layers

            student = GPT2LMHeadModel(config=student_config)

            # Copying all embeddings
            student.transformer.wte = teacher.transformer.wte
            student.transformer.wpe = teacher.transformer.wpe
            student.transformer.drop = teacher.transformer.drop
            # Maybe there is something else in BERT that need to be copied!
            # Specific thing for GPT2LMHead. Not necessary for BERT
            student.tie_weights()
            # Uniformly pick from middle layers from teacher
            # It is basically np.linspace(0, teacher_config.n_layer, num=student_config.n_layer, endpoint=True)
            step = (teacher_config.n_layer - 1) / (student_config.n_layer - 1)
            for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.n_layer)):
                student.transformer.h[student_layer] = teacher.transformer.h[teacher_layer]
        return student