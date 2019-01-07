import json

import dynet as dy
import numpy as np


class BERT(object):

    def __init__(self, bert_file, bert_mode):
        self.mode = bert_mode
        self.weights = []

        # Map the tokenized sentence to a list of the BERT layers
        self.sentence_to_layers = self._get_sentence_to_layers(bert_file)

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
        else:
            raise ValueError("BERT: Unsupported mode: %s" % self.mode)

    def get_sentence_representation(self, sentence):
        sentence_data = self.sentence_to_layers.get(sentence.lower())
        if not sentence_data:
            raise ValueError(
                "The sentence '%s' could not be found in the BERT data." \
                % sentence.lower()
            )

        return BERT.Sentence(sentence_data, self)

    def init_weights(self, model):
        # If a weighted average is computed, initialize the parameters
        if self.mode == "weighted_average":
            print "BERT: Learning a weighted average."
            self.weights = model.add_parameters(
                self.num_layers,
                name="bert-layer-weights",
                init="uniform",
                scale=1.0
            )

    def _get_sentence_to_layers(self, bert_file):
        print "Reading BERT embeddings from '%s'" % bert_file

        sentence_to_layers = {}
        for line in open(bert_file):
            if not line:
                continue

            sentence_data = json.loads(line)

            tokens = [
                feature["token"] for feature in sentence_data["features"]
            ]

            # Ignore the first and the last meta token
            tokens = tokens[1:-1]

            token_layers = []
            for token_data in sentence_data["features"]:
                if token_data["token"] in ["[SEP]", "[CLS]"]:  # TODO
                    # Ignore the tokens at beginning and end of the sentence
                    continue

                if token_data["token"].startswith("##"):
                    continue

                # Extract the layers for this token
                layers = [
                    layer["values"] for layer in token_data["layers"]
                ]

                layers = np.array(layers, dtype=float)
                token_layers.append(layers)

            sentence = " ".join(tokens)
            sentence = sentence.replace(" ##", "")
            print sentence
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
                return dy.inputTensor(np.sum(self.sentence_weights[i]))

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

            raise ValueError("BERT: Unsupported mode: %s" % self.bert.mode)
