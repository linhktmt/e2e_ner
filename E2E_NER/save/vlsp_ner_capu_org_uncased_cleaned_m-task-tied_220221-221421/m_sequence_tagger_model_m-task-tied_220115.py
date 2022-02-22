import logging
from pathlib import Path

import logging
from pathlib import Path
from typing import List, Union, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.datasets import SentenceDataset, StringDataset
import m_embeddings
from flair.file_utils import cached_path
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, : lens_[i]] = tensor

    return template, lens_


class SequenceTagger(flair.nn.Model):
    def __init__(
            self,
            hidden_size: int,
            rnn_input_dim: int,
            embedding_length: int,
            tag_dictionary: Dictionary,
            tag_capu_dictionary: Dictionary,
            tag_type: str,
            use_crf: bool = True,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            dropout: float = 0.8,
            word_dropout: float = 0.05,
            locked_dropout: float = 0.5,
            locked_dropout_capu: float = 0.8,

            capu_dropout: float = 0.9,
            sum_combine = False,
            linear_combine = True,
            train_initial_hidden_state: bool = False,
            rnn_type: str = "LSTM",
            pickle_module: str = "pickle",
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        """

        super(SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        # self.embeddings = embeddings
        self.rnn_input_dim = rnn_input_dim
        self.embedding_length = embedding_length

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_capu_dictionary: Dictionary = tag_capu_dictionary
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)
        self.tagset_capu_size: int = len(tag_capu_dictionary)
        self.capu_embedding_size: int = len(tag_capu_dictionary)
        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout
        self.use_capu_dropout: float = capu_dropout
        self.use_locked_dropout_capu: float = locked_dropout_capu
        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
            self.dropout_capu = torch.nn.Dropout(dropout)
        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
            self.locked_dropout_capu = flair.nn.LockedDropout(locked_dropout)
        if capu_dropout > 0.0:
            self.capu_dropout = nn.Dropout(capu_dropout)

        # rnn_input_dim: int = self.embeddings.embedding_length

        self.relearn_embeddings: bool = True

        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = rnn_type
        self.capu_embedding = nn.Embedding(self.capu_embedding_size, rnn_input_dim)
        self.sum_combine = sum_combine
        self.linear_combine = linear_combine
        self.capu_embed_learn = torch.nn.Linear(rnn_input_dim*2, rnn_input_dim)
        self.capu_lambda = 1
        # self.capu_drop = nn.Dropout(0.5)
        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    # self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    # TODO: Decide how to initialize the hidden state variables
                    # self.hs_initializer(self.lstm_init_h)
                    # self.hs_initializer(self.lstm_init_c)

            # final linear map to tag space
            self.linear = torch.nn.Linear(
                hidden_size * num_directions, len(tag_dictionary)
            )
        else:
            self.linear = torch.nn.Linear(
                # self.embeddings.embedding_length, len(tag_dictionary)
                embedding_length, len(tag_dictionary)
            )

        if self.use_crf:
            self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size, self.tagset_size) )

            self.transitions.detach()[self.tag_dictionary.get_idx_for_item(START_TAG), :] = -10000

            self.transitions.detach()[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000

        self.linear_capu = torch.nn.Linear(
            # self.embeddings.embedding_length, len(tag_dictionary)
            embedding_length, len(tag_capu_dictionary)
        )

        self.capu_embedding.weight = self.linear_capu.weight
        self.init_weights()
        '''
        ###    RNN for Capu task
        self.rnn_capu = getattr(torch.nn, self.rnn_type)(
            rnn_input_dim,
            hidden_size,
            num_layers=self.nlayers,
            dropout=0.0 if self.nlayers == 1 else 0.5,
            bidirectional=True,
            batch_first=True,
        )
        # Create initial hidden state and initialize it
        if self.train_initial_hidden_state:
            # self.hs_initializer = torch.nn.init.xavier_normal_

            self.lstm_init_h_capu = Parameter(torch.randn(self.nlayers * num_directions, self.hidden_size),
                requires_grad=True,
            )

            self.lstm_init_c_capu = Parameter(torch.randn(self.nlayers * num_directions, self.hidden_size),
                requires_grad=True,
            )

            # TODO: Decide how to initialize the hidden state variables
            # self.hs_initializer(self.lstm_init_h)
            # self.hs_initializer(self.lstm_init_c)
        '''
        # final linear map to tag space
        # self.linear_capu = torch.nn.Linear(hidden_size * num_directions, len(tag_capu_dictionary))
        if self.use_crf:
            self.transitions_capu = torch.nn.Parameter(torch.randn(self.tagset_capu_size, self.tagset_capu_size))

            self.transitions_capu.detach()[self.tag_capu_dictionary.get_idx_for_item(START_TAG), :] = -10000

            self.transitions_capu.detach()[:, self.tag_capu_dictionary.get_idx_for_item(STOP_TAG)] = -10000
        ### End RNN for capu task


        self.to(flair.device)

    def init_weights(self):
        initrange = 0.1
        self.capu_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_capu.bias.data.fill_(0)
        self.linear_capu.weight.data.uniform_(-initrange, initrange)


    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            # "embeddings": self.embeddings,
            "rnn_input_dim": self.rnn_input_dim,
            "embedding_length": self.embedding_length,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionary": self.tag_dictionary,
            "tag_capu_dictionary": self.tag_capu_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "use_capu_droput": self.use_capu_dropout,
            "sum_combine": self.sum_combine,
            "linear_combine": self.linear_combine,
            "rnn_type": self.rnn_type,
        }
        return model_state

    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if not "rnn_type" in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )
        use_capu_dropout = 0.0 if not "use_capu_dropout" in state.keys() else state["use_capu_dropout"]

        train_initial_hidden_state = (
            False
            if not "train_initial_hidden_state" in state.keys()
            else state["train_initial_hidden_state"]
        )

        model = SequenceTagger(
            hidden_size=state["hidden_size"],
            # embeddings=state["embeddings"],
            rnn_input_dim=state["rnn_input_dim"],
            embedding_length=state["embedding_length"],
            tag_dictionary=state["tag_dictionary"],
            tag_capu_dictionary=state["tag_capu_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            capu_dropout=use_capu_dropout,
            sum_combine=state["sum_combine"],
            linear_combine=state["linear_combine"],
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
        )
        model.load_state_dict(state["state_dict"])
        return model

    def predict(
            self,
            sentences: Union[List[Sentence], Sentence, List[str], str],
            mini_batch_size=32,
            embedding_storage_mode="none",
            all_tag_prob: bool = False,
            verbose: bool = False,
            use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
    ) -> List[Sentence]:
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a string or a List of Sentence or a List of string.
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param embedding_storage_mode: 'none' for the minimum memory footprint, 'cpu' to store embeddings in Ram,
        'gpu' to store embeddings in GPU memory.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param use_tokenizer: a custom tokenizer when string are provided (default is space based tokenizer).
        :return: List of Sentence enriched by the predicted tags
        """
        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence) or isinstance(sentences, str):
                sentences = [sentences]

            if (flair.device.type == "cuda") and embedding_storage_mode == "cpu":
                log.warning(
                    "You are inferring on GPU with parameter 'embedding_storage_mode' set to 'cpu'."
                    "This option will slow down your inference, usually 'none' (default value) "
                    "is a better choice."
                )

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )
            original_order_index = sorted(
                range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            if isinstance(sentences[0], Sentence):
                # remove previous embeddings
                store_embeddings(reordered_sentences, "none")
                dataset = SentenceDataset(reordered_sentences)
            else:
                dataset = StringDataset(
                    reordered_sentences, use_tokenizer=use_tokenizer
                )
            dataloader = DataLoader(
                dataset=dataset, batch_size=mini_batch_size, collate_fn=lambda x: x
            )

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            results: List[Sentence] = []
            for i, batch in enumerate(dataloader):

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {i}")
                results += batch
                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                feature: torch.Tensor = self.forward(batch)
                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(self.tag_type, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(self.tag_type, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            results: List[Union[Sentence, str]] = [
                results[index] for index in original_order_index
            ]
            assert len(sentences) == len(results)
            return results

    def evaluate(
            self,
            data_loader: DataLoader,
            out_path: Path = None,
            embedding_storage_mode: str = "none",
    ) -> (Result, float):

        if type(out_path) == str:
            out_path = Path(out_path)

        with torch.no_grad():
            eval_loss = 0
            eval_loss_capu = 0
            batch_no: int = 0

            metric = Metric("Evaluation")

            lines: List[str] = []

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features, features_capu = self.forward(batch)
                    loss, loss_capu = self._calculate_loss(features, features_capu, batch)
                    tags, _ = self._obtain_labels(
                        feature=features,
                        batch_sentences=batch,
                        transitions=transitions,
                        get_all_tags=False,
                    )

                eval_loss += loss
                eval_loss_capu += loss_capu

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                store_embeddings(batch, embedding_storage_mode)

            eval_loss /= batch_no
            eval_loss_capu /= batch_no
            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss, eval_loss_capu

    def forward_loss(self, data_points: Union[List[Sentence], Sentence], sort=True) -> torch.tensor:
        features, features_capu = self.forward(data_points)
        return self._calculate_loss(features, features_capu, data_points)

    @staticmethod
    def get_sentence_tensor(sentences: List[Sentence]):
        embedding_model = m_embeddings.get_trained_embedding()
        embedding_model.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            embedding_model.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : embedding_model.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                embedding_model.embedding_length,
            ]
        )
        return sentence_tensor, lengths, sentences

    def forward(self, sentences: List[Sentence]):
        sentence_tensor, lengths, sentences = SequenceTagger.get_sentence_tensor(sentences)
        # self.embeddings.embed(sentences)
        #
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        #
        pre_allocated_zero_tensor = torch.zeros((longest_token_sequence_in_batch, self.capu_embedding_size),
            dtype=torch.float,
            device=flair.device,
        )

        ###### Get capu tag for embedding ##################
        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_capu_dictionary.get_idx_for_item(token.get_tag('capu').value) #TODO: tag_type = 'capu'
                for token in sentence
            ]

            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
            if nb_padding_tokens > 0:
                # t = pre_allocated_zero_tensor[: self.capu_embedding_size * nb_padding_tokens]
                t = [0] * nb_padding_tokens
                tag_idx.extend(t)
            # add tags as tensor
            # tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag_idx)
        # tag_tensor = torch.FloatTensor(tag_list)
        tag_tensor = torch.as_tensor(tag_list, device=flair.device)
        tag_embedded = self.capu_embedding(tag_tensor)
        if self.use_capu_dropout > 0.0:
            tag_embedded = self.capu_dropout(tag_embedded)
        if self.sum_combine:
            sentence_tensor = sentence_tensor + tag_embedded
        if self.linear_combine:
            sentence_tensor = self.capu_embed_learn(torch.cat((sentence_tensor, tag_embedded), 2))
        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0: # (batch_size, max_len, embedded_dim)
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)
        features = self.linear(sentence_tensor) #TODO: Add Log_softmax

        '''
        ##### RNN for capu task ###############
        packed_capu = torch.nn.utils.rnn.pack_padded_sequence(
            tag_embedded, lengths, enforce_sorted=False, batch_first=True
        )

        # if initial hidden state is trainable, use this state
        if self.train_initial_hidden_state:
            initial_hidden_state_capu = [
                self.lstm_init_h_capu.unsqueeze(1).repeat(1, len(sentences), 1),
                self.lstm_init_c_capu.unsqueeze(1).repeat(1, len(sentences), 1),
            ]
            rnn_output_capu, hidden_capu = self.rnn(packed_capu, initial_hidden_state_capu)
        else:
            rnn_output_capu, hidden_capu = self.rnn(packed_capu)

        sentence_tensor_capu, output_lengths_capu = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_output_capu, batch_first=True
        )
        
        if self. > 0.0:
            sentence_tensor_capu = self.dropout_capu(sentence_tensor_capu)
        # word dropout only before LSTM - TODO: more experimentation needed
        # if self.use_wouse_dropoutrd_dropout > 0.0:
        #     sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout_capu > 0.0:
            sentence_tensor_capu = self.locked_dropout_capu(sentence_tensor_capu)
        '''
        if self.use_dropout > 0.0:
            tag_embedded = self.dropout_capu(tag_embedded)
        features_capu = self.linear_capu(tag_embedded)


        return features, features_capu # (b_size, max_len, #tag)

    def _score_sentence(self, feats, tags, lens_, tag_type = 'ner'):
        if tag_type=='ner':
            start = torch.tensor(
                [self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
            )
            start = start[None, :].repeat(tags.shape[0], 1)

            stop = torch.tensor(
                [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
            )
            stop = stop[None, :].repeat(tags.shape[0], 1)

            pad_start_tags = torch.cat([start, tags], 1)
            pad_stop_tags = torch.cat([tags, stop], 1)

            for i in range(len(lens_)):
                pad_stop_tags[i, lens_[i]:] = self.tag_dictionary.get_idx_for_item(
                    STOP_TAG
                )

            score = torch.FloatTensor(feats.shape[0]).to(flair.device)

            for i in range(feats.shape[0]):
                r = torch.LongTensor(range(lens_[i])).to(flair.device)

                score[i] = torch.sum(
                    self.transitions[
                        pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                    ]
                ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])
        else:
            start = torch.tensor([self.tag_capu_dictionary.get_idx_for_item(START_TAG)], device=flair.device)
            start = start[None, :].repeat(tags.shape[0], 1)

            stop = torch.tensor([self.tag_capu_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device)
            stop = stop[None, :].repeat(tags.shape[0], 1)

            pad_start_tags = torch.cat([start, tags], 1)
            pad_stop_tags = torch.cat([tags, stop], 1)

            for i in range(len(lens_)):
                pad_stop_tags[i, lens_[i]:] = self.tag_capu_dictionary.get_idx_for_item(STOP_TAG)

            score = torch.FloatTensor(feats.shape[0]).to(flair.device)

            for i in range(feats.shape[0]):
                r = torch.LongTensor(range(lens_[i])).to(flair.device)

                score[i] = torch.sum(
                    self.transitions_capu[pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]]) \
                           + torch.sum(feats[i, r, tags[i, : lens_[i]]])  # TODO: CUDA error: device-side assert triggered


        return score



    def _calculate_loss(self, features: torch.tensor, features_capu: torch.tensor, sentences: List[Sentence]) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        ### capu ###
        tag_capu_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_capu_idx: List[int] = [
                self.tag_capu_dictionary.get_idx_for_item(token.get_tag('capu').value)
                for token in sentence
            ]
            # add tags as tensor
            tag_capu = torch.tensor(tag_capu_idx, device=flair.device)
            tag_capu_list.append(tag_capu)

        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = pad_tensors(tag_list)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            ### capu ###
            tags_capu, _ = pad_tensors(tag_capu_list)

            forward_capu_score = self._forward_alg(features_capu, lengths, capu=True)
            gold_capu_score = self._score_sentence(features_capu, tags_capu, lengths, tag_type='capu')

            score_capu = forward_capu_score - gold_capu_score
            score_sum = score.mean() + score_capu.mean()
            return score.mean(), score_capu.mean()*self.capu_lambda  #TODO: lambda analysis
            # return score_sum

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                    features, tag_list, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]

                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags
                )
            score /= len(features)
            return score

    def _obtain_labels(
            self,
            feature: torch.Tensor,
            batch_sentences: List[Sentence],
            transitions: Optional[np.ndarray],
            get_all_tags: bool,
    ) -> (List[List[Label]], List[List[List[Label]]]):
        """
        Returns a tuple of two lists:
         - The first list corresponds to the most likely `Label` per token in each sentence.
         - The second list contains a probability distribution over all `Labels` for each token
           in a sentence for all sentences.
        """

        lengths: List[int] = [len(sentence.tokens) for sentence in batch_sentences]

        tags = []
        all_tags = []
        feature = feature.cpu()
        if self.use_crf:
            feature = feature.numpy()
        else:
            for index, length in enumerate(lengths):
                feature[index, length:] = 0
            softmax_batch = F.softmax(feature, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feature = zip(softmax_batch, scores_batch, prediction_batch)

        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(
                    feats=feats[:length],
                    transitions=transitions,
                    all_scores=get_all_tags,
                )
            else:
                softmax, score, prediction = feats
                confidences = score[:length].tolist()
                tag_seq = prediction[:length].tolist()
                scores = softmax[:length].tolist()

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            if get_all_tags:
                all_tags.append(
                    [
                        [
                            Label(
                                self.tag_dictionary.get_item_for_index(score_id), score
                            )
                            for score_id, score in enumerate(score_dist)
                        ]
                        for score_dist in scores
                    ]
                )

        return tags, all_tags

    @staticmethod
    def _softmax(x, axis):
        # reduce raw values to avoid NaN during exp
        x_norm = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x_norm)
        return y / y.sum(axis=axis, keepdims=True)

    def _viterbi_decode(
            self, feats: np.ndarray, transitions: np.ndarray, all_scores: bool
    ):
        id_start = self.tag_dictionary.get_idx_for_item(START_TAG)
        id_stop = self.tag_dictionary.get_idx_for_item(STOP_TAG)

        backpointers = np.empty(shape=(feats.shape[0], self.tagset_size), dtype=np.int_)
        backscores = np.empty(
            shape=(feats.shape[0], self.tagset_size), dtype=np.float32
        )

        init_vvars = np.expand_dims(
            np.repeat(-10000.0, self.tagset_size), axis=0
        ).astype(np.float32)
        init_vvars[0][id_start] = 0

        forward_var = init_vvars
        for index, feat in enumerate(feats):
            # broadcasting will do the job of reshaping and is more efficient than calling repeat
            next_tag_var = forward_var + transitions
            bptrs_t = next_tag_var.argmax(axis=1)
            viterbivars_t = next_tag_var[np.arange(bptrs_t.shape[0]), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores[index] = forward_var
            forward_var = forward_var[np.newaxis, :]
            backpointers[index] = bptrs_t

        terminal_var = forward_var.squeeze() + transitions[id_stop]
        terminal_var[id_stop] = -10000.0
        terminal_var[id_start] = -10000.0
        best_tag_id = terminal_var.argmax()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == id_start
        best_path.reverse()

        best_scores_softmax = self._softmax(backscores, axis=1)
        best_scores_np = np.max(best_scores_softmax, axis=1)

        # default value
        all_scores_np = np.zeros(0, dtype=np.float64)
        if all_scores:
            all_scores_np = best_scores_softmax
            for index, (tag_id, tag_scores) in enumerate(zip(best_path, all_scores_np)):
                if type(tag_id) != int and tag_id.item() != tag_scores.argmax():
                    swap_index_score = tag_scores.argmax()
                    all_scores_np[index][tag_id.item()], all_scores_np[index][
                        swap_index_score
                    ] = (
                        all_scores_np[index][swap_index_score],
                        all_scores_np[index][tag_id.item()],
                    )
                elif type(tag_id) == int and tag_id != tag_scores.argmax():
                    swap_index_score = tag_scores.argmax()
                    all_scores_np[index][tag_id], all_scores_np[index][
                        swap_index_score
                    ] = (
                        all_scores_np[index][swap_index_score],
                        all_scores_np[index][tag_id],
                    )

        return best_scores_np.tolist(), best_path, all_scores_np.tolist()

    def _forward_alg(self, feats, lens_, capu = False):

        if not capu:
            init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
            init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0
        else:
            init_alphas = torch.FloatTensor(self.tagset_capu_size).fill_(-10000.0)
            init_alphas[self.tag_capu_dictionary.get_idx_for_item(START_TAG)] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=flair.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        if not capu:
            transitions = self.transitions.view(
                1, self.transitions.shape[0], self.transitions.shape[1]
            ).repeat(feats.shape[0], 1, 1)
        else:
            transitions = self.transitions_capu.view(
                1, self.transitions_capu.shape[0], self.transitions_capu.shape[1]
            ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        if not capu:
            terminal_var = forward_var + self.transitions[
                                             self.tag_dictionary.get_idx_for_item(STOP_TAG)
                                         ][None, :].repeat(forward_var.shape[0], 1)
        else:
            terminal_var = forward_var + self.transitions_capu[
                                             self.tag_capu_dictionary.get_idx_for_item(STOP_TAG)
                                         ][None, :].repeat(forward_var.shape[0], 1)
        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def _forward_alg_capu(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_cpu_dictionary.get_idx_for_item(START_TAG)] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=flair.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions_capu.view(
            1, self.transitions_capu.shape[0], self.transitions_capu.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
                                         self.tag_dictionary.get_idx_for_item(STOP_TAG)
                                     ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha
    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    @staticmethod
    def _filter_empty_string(texts: List[str]) -> List[str]:
        filtered_texts = [text for text in texts if text]
        if len(texts) != len(filtered_texts):
            log.warning(
                f"Ignore {len(texts) - len(filtered_texts)} string(s) with no tokens."
            )
        return filtered_texts

    def _fetch_model(model_name) -> str:

        model_map = {}

        aws_resource_path_v04 = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4"
        )

        model_map["ner"] = "/".join(
            [aws_resource_path_v04, "NER-conll03-english", "en-ner-conll03-v0.4.pt"]
        )

        model_map["ner-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "NER-conll03--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward-fast%2Bnews-backward-fast-normal-locked0.5-word0.05--release_4",
                "en-ner-fast-conll03-v0.4.pt",
            ]
        )

        model_map["ner-ontonotes"] = "/".join(
            [
                aws_resource_path_v04,
                "release-ner-ontonotes-0",
                "en-ner-ontonotes-v0.4.pt",
            ]
        )

        model_map["ner-ontonotes-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-ner-ontonotes-fast-0",
                "en-ner-ontonotes-fast-v0.4.pt",
            ]
        )

        for key in ["ner-multi", "multi-ner"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "release-quadner-512-l2-multi-embed",
                    "quadner-large.pt",
                ]
            )

        for key in ["ner-multi-fast", "multi-ner-fast"]:
            model_map[key] = "/".join(
                [aws_resource_path_v04, "NER-multi-fast", "ner-multi-fast.pt"]
            )

        for key in ["ner-multi-fast-learn", "multi-ner-fast-learn"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "NER-multi-fast-evolve",
                    "ner-multi-fast-learn.pt",
                ]
            )

        model_map["pos"] = "/".join(
            [
                aws_resource_path_v04,
                "POS-ontonotes--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
                "en-pos-ontonotes-v0.4.pt",
            ]
        )

        model_map["pos-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-pos-fast-0",
                "en-pos-ontonotes-fast-v0.4.pt",
            ]
        )

        for key in ["pos-multi", "multi-pos"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "release-dodekapos-512-l2-multi",
                    "pos-multi-v0.1.pt",
                ]
            )

        for key in ["pos-multi-fast", "multi-pos-fast"]:
            model_map[key] = "/".join(
                [aws_resource_path_v04, "UPOS-multi-fast", "pos-multi-fast.pt"]
            )

        model_map["frame"] = "/".join(
            [aws_resource_path_v04, "release-frame-1", "en-frame-ontonotes-v0.4.pt"]
        )

        model_map["frame-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-frame-fast-0",
                "en-frame-ontonotes-fast-v0.4.pt",
            ]
        )

        model_map["chunk"] = "/".join(
            [
                aws_resource_path_v04,
                "NP-conll2000--h256-l1-b32-p3-0.5-%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
                "en-chunk-conll2000-v0.4.pt",
            ]
        )

        model_map["chunk-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-chunk-fast-0",
                "en-chunk-conll2000-fast-v0.4.pt",
            ]
        )

        model_map["da-pos"] = "/".join(
            [aws_resource_path_v04, "POS-danish", "da-pos-v0.1.pt"]
        )

        model_map["da-ner"] = "/".join(
            [aws_resource_path_v04, "NER-danish", "da-ner-v0.1.pt"]
        )

        model_map["de-pos"] = "/".join(
            [aws_resource_path_v04, "release-de-pos-0", "de-pos-ud-hdt-v0.4.pt"]
        )

        model_map["de-pos-fine-grained"] = "/".join(
            [
                aws_resource_path_v04,
                "POS-fine-grained-german-tweets",
                "de-pos-twitter-v0.1.pt",
            ]
        )

        model_map["de-ner"] = "/".join(
            [aws_resource_path_v04, "release-de-ner-0", "de-ner-conll03-v0.4.pt"]
        )

        model_map["de-ner-germeval"] = "/".join(
            [aws_resource_path_v04, "NER-germeval", "de-ner-germeval-0.4.1.pt"]
        )

        model_map["fr-ner"] = "/".join(
            [aws_resource_path_v04, "release-fr-ner-0", "fr-ner-wikiner-0.4.pt"]
        )
        model_map["nl-ner"] = "/".join(
            [aws_resource_path_v04, "NER-conll2002-dutch", "nl-ner-conll02-v0.1.pt"]
        )

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name

    def get_transition_matrix(self):
        data = []
        for to_idx, row in enumerate(self.transitions):
            for from_idx, column in enumerate(row):
                row = [
                    self.tag_dictionary.get_item_for_index(from_idx),
                    self.tag_dictionary.get_item_for_index(to_idx),
                    column.item(),
                ]
                data.append(row)
            data.append(["----"])
        print(tabulate(data, headers=["FROM", "TO", "SCORE"]))
