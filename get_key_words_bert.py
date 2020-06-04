from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import os
import pprint
import numpy as np
import tensorflow as tf
import pickle

from bert import tokenization
from bert import modeling
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re, string
from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib

def preprocessing(text):
    russian_stopwords = stopwords.words("russian")
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    stemmer = SnowballStemmer("russian")
    text = regex.sub('', text)
    text = [token for token in text.split() if token not in russian_stopwords]
    text = [stemmer.stem(token) for token in text]
    text = [token for token in text if token]
    return ' '.join(text)

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        
class InputFeatures(object):
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def input_fn_builder(features, seq_length):
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)
    
    def input_fn(params):
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
            "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
            "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn

def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):  
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
             tvars, init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
        "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn

def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def read_sequence(input_sentences):
    examples = []
    unique_id = 0
    for sentence in input_sentences:
        line = tokenization.convert_to_unicode(sentence)
        examples.append(InputExample(unique_id=unique_id, text_a=line))
        unique_id += 1
    return examples

def get_features(input_text, dim=768):
    layer_indexes = [-1,-2,-3,-4]

    bert_config = modeling.BertConfig.from_json_file('multilingual_L-12_H-768_A-12' + '/bert_config.json')

    tokenizer = tokenization.FullTokenizer(
        vocab_file='multilingual_L-12_H-768_A-12' + '/vocab.txt', do_lower_case=True)

    run_config = tf.contrib.tpu.RunConfig()

    examples = read_sequence(input_text)

    features = convert_examples_to_features(
        examples=examples, seq_length=87, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint='multilingual_L-12_H-768_A-12' + '/bert_model.ckpt',
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=True)
    
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    
    run_config = tf.contrib.tpu.RunConfig(
      master=None,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=8,
          per_host_input_for_training=is_per_host))
    
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=128)

    input_fn = input_fn_builder(
          features=features, seq_length=87)


    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        output = collections.OrderedDict()
        for (i, token) in enumerate(feature.tokens):
            layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layer_output_flat = np.array([x for x in layer_output[i:(i + 1)].flat])
                layers.append(layer_output_flat)
            output[token] = sum(layers)[:dim]
  
    return output

def embeddings (text):
    text_bert = preprocessing(text)
    embeddings = get_features(text_bert)
    del embeddings['[CLS]']
    del embeddings['[SEP]']
    A = []
    for k in embeddings.keys():
        A.extend(embeddings[k])
    
    return A

def get_key_words_bert(text,n_topics,n_keywords):
    A = embeddings(text)
    lda = joblib.load('lda1.pkl')
    count_vect = joblib.load('countVect.pkl')
    lda_topics = [A[100:249]]
    lda_topics = np.squeeze(lda_topics, axis=0)
    n_topics_indices = lda_topics.argsort()[-n_topics:][::-1]
    top_topics_words_dists = []
    for i in n_topics_indices:
        top_topics_words_dists.append(lda.components_[i])
        
    keywords = np.zeros(shape=(n_keywords*n_topics, lda.components_.shape[1]))
    for i,topic in enumerate(top_topics_words_dists):
        n_keywords_indices = topic.argsort()[-n_keywords:][::-1]
        for k,j in enumerate(n_keywords_indices):
            keywords[i * n_keywords + k, j] = 1
    keywords = count_vect.inverse_transform(keywords)
    keywords = [keyword[0] for keyword in keywords]
    return keywords