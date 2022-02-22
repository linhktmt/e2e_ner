import logging
from typing import List, Dict
import torch

from flair.embeddings import ScalarMix
import flair
from flair.data import Token, Sentence

from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from fairseq.models.roberta import XLMRModel

log = logging.getLogger("flair")

embedding = None


def extract_embeddings(
        hidden_states,
        pooling_operation: str,
        subword_start_idx: int,
        subword_end_idx: int,
        use_scalar_mix: bool = False,
) -> List[torch.FloatTensor]:
    """
    Extracts subword embeddings from specified layers from hidden states.
    :param hidden_states: list of hidden states from model
    :param layers: list of layers
    :param pooling_operation: pooling operation for subword embeddings (supported: first, last, first_last and mean)
    :param subword_start_idx: defines start index for subword
    :param subword_end_idx: defines end index for subword
    :param use_scalar_mix: determines, if scalar mix should be used
    :return: list of extracted subword embeddings
    """
    subtoken_embeddings: List[torch.FloatTensor] = []

    current_embeddings = hidden_states[0][subword_start_idx:subword_end_idx]

    first_embedding = current_embeddings[0]

    if pooling_operation == "first_last":
        last_embedding = current_embeddings[-1]
        final_embedding = torch.cat(
            [first_embedding, last_embedding]
        )
    elif pooling_operation == "last":
        final_embedding = current_embeddings[-1]
    elif pooling_operation == "mean":
        all_embeddings = [
            embedding.unsqueeze(0) for embedding in current_embeddings
        ]
        final_embedding = torch.mean(
            torch.cat(all_embeddings, dim=0), dim=0
        )
    else:
        final_embedding = first_embedding

    subtoken_embeddings.append(final_embedding)

    if use_scalar_mix:
        sm = ScalarMix(mixture_size=len(subtoken_embeddings))
        sm_embeddings = sm(subtoken_embeddings)

        subtoken_embeddings = [sm_embeddings]

    return subtoken_embeddings


def get_xlmr_sentence_embeddings(
        sentences: List[Sentence],
        model: XLMRModel,
        name: str,
        pooling_operation: str,
        use_scalar_mix: bool
) -> List[Sentence]:
    """
    Builds sentence embeddings for Transformer-based architectures.
    :param sentences: input sentences
    :param model: model object
    :param name: name of the Transformer-based model
    :param pooling_operation: defines pooling operation for subword extraction
    :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
    :return: list of sentences (each token of a sentence is now embedded)
    """
    with torch.no_grad():
        for sentence in sentences:
            sentence_text = " ".join([t.text for t in sentence.tokens])
            token_subwords_mapping: Dict[int, int] = {}

            for token in sentence.tokens:
                token_text = token.text

                subwords = model.bpe.sp.EncodeAsPieces(token_text)

                token_subwords_mapping[token.idx] = len(subwords)

            offset = 1  # Offset ignore start token

            indexed_tokens = model.encode(sentence_text)
            hidden_states = model.extract_features(indexed_tokens)

            for token in sentence.tokens:
                len_subwords = token_subwords_mapping[token.idx]

                try:
                    subtoken_embeddings = extract_embeddings(
                        hidden_states=hidden_states,
                        pooling_operation=pooling_operation,
                        subword_start_idx=offset,
                        subword_end_idx=offset + len_subwords,
                        use_scalar_mix=use_scalar_mix,
                    )
                except:
                    print(sentence)

                offset += len_subwords

                final_subtoken_embedding = torch.cat(subtoken_embeddings)
                token.set_embedding(name, final_subtoken_embedding)

    return sentences


class XLMREmbeddings(TokenEmbeddings):
    def __init__(
            self,
            model_name_or_path: str = "./model",
            pooling_operation: str = "mean",
            use_scalar_mix: bool = False,
    ):
        """
        XLM embeddings, as proposed in Guillaume et al., 2019.
        :param model_name_or_path: name or path of XLMR model
        :param pooling_operation: defines pooling operation for subwords
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()

        self.name = 'XLMREmbeddings'
        self.model_xlmr = XLMRModel.from_pretrained(model_name_or_path)
        self.model = self.model_xlmr.model
        print(self.model)
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True
        self.model_xlmr.to(flair.device)
        self.model_xlmr.eval()
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        sentences = get_xlmr_sentence_embeddings(
            sentences=sentences,
            model=self.model_xlmr,
            name=self.name,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix
        )

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name


def get_trained_embedding(model_name_or_path=None):
    global embedding
    if embedding is None:
        # 4. initialize embeddings
        embedding_types: List[TokenEmbeddings] = [
            XLMREmbeddings(model_name_or_path=model_name_or_path),
            # comment in these lines to use flair embeddings
            # FlairEmbeddings('news-forward'),
            # FlairEmbeddings('news-backward'),
        ]

        embedding = StackedEmbeddings(embeddings=embedding_types)
    return embedding
