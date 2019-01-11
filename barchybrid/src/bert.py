import json

import dynet as dy
import numpy as np


class BERT(object):

    def __init__(self, bert_file, token_mapping_file, bert_mode,
                 multitoken_selection_strategy, output_layer_size):
        self.mode = bert_mode

        # Initialize parameters that might be used depending on the mode.
        self.weights, self.finetune_w1, self.finetune_b1 = None, None, None
        self.finetune_activation = dy.rectify

        # Map the tokenized sentence to a list of the BERT layers
        self.sentence_to_layers = self._get_sentence_to_layers(
            bert_file, token_mapping_file, multitoken_selection_strategy)

        # Determine the numbers of layers and their dimensions
        tokens = next(self.sentence_to_layers.itervalues())
        self.num_layers = len(tokens[0])
        self.emb_dim = len(tokens[0][0])

        # Depending on the mode, the output will have different dimensions
        if self.mode == "concatenate":
            self.output_dim = self.emb_dim * self.num_layers
        elif self.mode == "weighted_average":
            self.output_dim = self.emb_dim
        elif self.mode == "sum":
            self.output_dim = self.emb_dim
        elif self.mode == "finetune":
            self.output_dim = output_layer_size
        else:
            raise ValueError("BERT: Unsupported mode: %s" % self.mode)

    def get_sentence_representation(self, sentence):
        sentence_data = self.sentence_to_layers.get(sentence)
        if not sentence_data:
            raise ValueError(
                "The sentence '%s' could not be found in the BERT data." \
                % sentence
            )

        return BERT.Sentence(sentence_data, self)

    def init_weights(self, model):
        if self.mode == "weighted_average":
            print "BERT: Learning a weighted average."
            self.weights = model.add_parameters(
                self.num_layers,
                name="bert-layer-weights",
                init="uniform",
                scale=1.0
            )

        if self.mode == "finetune":
            print "BERT: Fine-tuning via separate layer."
            self.finetune_w1 = model.add_parameters(
                (self.output_dim, self.emb_dim * self.num_layers),
                name="bert-finetune-W1"
            )
            self.finetune_b1 = model.add_parameters(
                self.output_dim,
                name="bert-finetune-b1"
            )

    def _get_sentence_to_layers(self, bert_file, mapping_file,
                                multitoken_selection_strategy):
        print "Reading BERT embeddings from '%s'" % bert_file

        sentence_to_layers = {}
        for bert_raw, mapping_raw in zip(open(bert_file), open(mapping_file)):
            sentence_data = json.loads(bert_raw)
            token_mapping = json.loads(mapping_raw)

            # Gold segmented sentence from the treebank
            sentence = token_mapping["sentence"]

            # Check that this is the right mapping for the sentence
            bert_sentence = " ".join(
                [item["token"] for item in sentence_data["features"]]
            )

            assert token_mapping["bert_sentence"] == bert_sentence, \
                "BERT sentence missmatch. Is the mapping file correct?"

            token_layers = []
            num_gold_tokens = len(token_mapping["token_map"])
            num_bert_tokens = len(sentence_data["features"])

            # Select token layers based on mapping
            for i, token_index in enumerate(token_mapping["token_map"]):
                # Check if for this gold token, there are multiple
                # BERT tokens
                token_span_length = 1
                if i == num_gold_tokens - 1:
                    # This is the last token.
                    token_span_length = num_bert_tokens - token_index - 1
                else:
                    # Look at the index of the next gold token to check for
                    # multi tokens.
                    next_index = token_mapping["token_map"][i + 1]
                    if next_index > token_index + 1:
                        token_span_length = next_index - token_index

                if token_span_length > 1:
                    token_data = {}
                    # BERT has multiple representations for this gold token
                    if multitoken_selection_strategy in ["first", "last"]:
                        if multitoken_selection_strategy == "first":
                            token_data = sentence_data["features"][token_index]

                        elif multitoken_selection_strategy == "last":
                            index = token_index + token_span_length - 1
                            token_data = sentence_data["features"][index]

                        layers = [
                            layer["values"] for layer in token_data["layers"]
                        ]

                    elif multitoken_selection_strategy == "average":
                        tokens = sentence_data["features"] \
                            [token_index:token_index + token_span_length]
                        layers = [
                            [layer["values"] for layer in token["layers"]]
                            for token in tokens
                        ]
                        layers = np.mean(np.array(layers, dtype=float), axis=0)

                    else:
                        raise ValueError(
                            "Invalid BERT multitoken selection strategy: '%s'"
                            % multitoken_selection_strategy)
                else:
                    token_data = sentence_data["features"][token_index]
                    layers = [
                        layer["values"] for layer in token_data["layers"]
                    ]

                layers = np.array(layers, dtype=float)
                token_layers.append(layers)

            sentence_to_layers[sentence] = token_layers

        return sentence_to_layers

    class Sentence(object):

        def __init__(self, sentence_weights, bert):
            self.sentence_weights = sentence_weights
            self.bert = bert

        def __getitem__(self, i):
            """
            Return the layer for the current word.
            :param i: Word at index i in the sentence.
            :return: Embedding for the word
            """
            if self.bert.mode == "concatenate":
                return dy.inputTensor(self.sentence_weights[i].flatten())

            elif self.bert.mode == "sum":
                return dy.inputTensor(
                    np.sum(np.array(self.sentence_weights[i])))

            elif self.bert.mode == "weighted_average":
                normalized_weights = dy.softmax(self.bert.weights)
                y_hat = [
                    dy.inputTensor(layer) * weight
                    for layer, weight in zip(
                        self.sentence_weights[i],
                        normalized_weights
                    )
                ]

                return dy.esum(y_hat)

            elif self.bert.mode == "finetune":
                x = dy.inputTensor(self.sentence_weights[i].flatten())
                return self.bert.finetune_activation(
                    self.bert.finetune_w1 * x + self.bert.finetune_b1
                )

            raise ValueError("BERT: Unsupported mode: %s" % self.bert.mode)
