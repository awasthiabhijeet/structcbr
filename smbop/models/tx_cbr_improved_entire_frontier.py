import itertools
import json
import logging
import os
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict

import allennlp
import torch
#torch.autograd.set_detect_anomaly(True)
from allennlp.common.util import *
from allennlp.data import TokenIndexer, Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2SeqEncoder,
    TextFieldEmbedder,
)

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.nn.util import masked_mean, masked_softmax
from allennlp.training.metrics import Average
from anytree import PostOrderIter
from overrides import overrides

import smbop.utils.node_util as node_util
from smbop.eval_final.evaluation import evaluate_single #returns exact match
from smbop.utils import ra_postproc
from smbop.utils import vec_utils
from smbop.utils import hashing
from smbop.modules.lxmert import LxmertCrossAttentionLayer

import numpy as np

from torch_scatter import scatter_logsumexp, scatter_add, scatter_mean

logger = logging.getLogger(__name__)

MAX_G1 = 0
MAX_G2 = 0

from allennlp.modules.seq2seq_encoders import PytorchTransformer

@Model.register("tx_cbr_improved_entire_frontier")
class TXCBRImprovedEntireFrontier(Model):
    '''
    All the init arguments are probably loaded from the json config file
    '''
    def __init__(
        self,
        experiment_name: str,
        vocab: Vocabulary,
        question_embedder: TextFieldEmbedder, #grappa etc. (type: pytorch_transformer)
        schema_encoder: Seq2SeqEncoder, # (type: relation transformer) [RAT layers?]
        beam_encoder: Seq2SeqEncoder, # (type: pytorch_transformer) [Used for Contextualizing beam w.r.t. inputs]
        tree_rep_transformer: Seq2SeqEncoder, # (type: pytorch_transformer)
        utterance_augmenter: Seq2SeqEncoder, # (type: cross_attention)
        beam_summarizer: Seq2SeqEncoder, # (type: pytorch_transformer) # not used anywhere
        decoder_timesteps=9,
        beam_size=30,
        misc_params=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)
        self._experiment_name = experiment_name
        self._misc_params = misc_params
        self.set_flags()
        self._utterance_augmenter = utterance_augmenter
        self._action_dim = beam_encoder.get_output_dim()
        self._beam_size = beam_size
        self._n_schema_leafs = 15
        self._num_values = 10

        self.tokenizer = TokenIndexer.by_name("pretrained_transformer")( # hardcoding
            model_name="roberta-base"
        )._allennlp_tokenizer.tokenizer

        if not self.cntx_reranker:
            self._noreranker_cntx_linear = torch.nn.Linear( # not used anywhere ?
                in_features=self._action_dim, out_features=2 * self._action_dim
            )
        if not self.utt_aug:
            self._nobeam_cntx_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2 * self._action_dim
            )
        self.activation_func = torch.nn.ReLU
        if self.lin_after_cntx:
            self.cntx_linear = torch.nn.Sequential(
                torch.nn.Linear(2 * self._action_dim, 4 * self._action_dim),
                torch.nn.Dropout(p=dropout),
                torch.nn.LayerNorm(4 * self._action_dim),
                self.activation_func(),
                torch.nn.Linear(4 * self._action_dim, 2 * self._action_dim),
            )
        if self.cntx_rep:
            self._cntx_rep_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2 * self._action_dim
            )
        self._create_action_dicts() # defines ops and frontier size
        self.op_count = self.binary_op_count + self.unary_op_count #total op count
        self.xent = torch.nn.CrossEntropyLoss() # not used anywhere ?

        self.type_embedding = torch.nn.Embedding(self.op_count, self._action_dim) #op embedding?
        self.summrize_vec = torch.nn.Embedding(
            num_embeddings=1, embedding_dim=self._action_dim
        ) #? not used anywhere?

        self.d_frontier = 2 * self._action_dim
        self.left_emb = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.d_frontier
        ) #?
        self.right_emb = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.d_frontier
        ) #?
        self.after_add = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
            torch.nn.Linear(self.d_frontier, self.d_frontier),
        )
        self._unary_frontier_embedder = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
            torch.nn.Linear(self.d_frontier, self.d_frontier),
        )

        self.op_linear = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.op_count
        )
        self.pre_op_linear = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
        )

        assert (self._action_dim % 2) == 0
        self.vocab = vocab
        self._question_embedder = question_embedder
        self._schema_encoder = schema_encoder
        self._beam_encoder = beam_encoder
        self._beam_summarizer = beam_summarizer #not used anywhere ?

        self._tree_rep_transformer = tree_rep_transformer

        self._decoder_timesteps = decoder_timesteps
        self._beam_size = beam_size
        self.q_emb_dim = question_embedder.get_output_dim()

        self.dropout_prob = dropout
        self._action_dim = beam_encoder.get_output_dim()
        self._span_score_func = torch.nn.Linear(self._action_dim, 2)
        self._pooler = BagOfEmbeddingsEncoder(embedding_dim=self._action_dim)

        self._rank_schema = torch.nn.Sequential(
            torch.nn.Linear(self._action_dim, self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self._action_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._action_dim, 1),
        )
        self._rank_beam = torch.nn.Sequential( # not used anywhere ?
            torch.nn.Linear(2 * self._action_dim, 2 * self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(2 * self._action_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * self._action_dim, 1),
        )
        self._emb_to_action_dim = torch.nn.Linear( # used in _encode_utt_schema
            in_features=self.q_emb_dim,
            out_features=self._action_dim,
        )

        self._create_type_tensor() # used in typecheck_frontier

        self._bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none") # not used anywhere?

        self._softmax = torch.nn.Softmax(dim=1) # not used anywhere
        self._final_beam_acc = Average()
        self._reranker_acc = Average()
        self._spider_acc = Average()

        self._leafs_acc = Average()
        self._batch_size = -1 # NOT USED ANYWHERE ELSE IN THIS FILE 
        self._device = None
        self._evaluate_func = partial(
            evaluate_single, #returns exact match
            db_dir=os.path.join("dataset", "database"), # hardcoding
            table_file=os.path.join("dataset", "tables.json"), # hardcoding
        )

        ###### New Params ########

        self.norm_beam_sum = torch.nn.LayerNorm(2*self._action_dim)
        self.ff_combo = torch.nn.Sequential(
                    torch.nn.Linear(2*self._action_dim, 4*self._action_dim),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.LayerNorm(4*self._action_dim),
                    self.activation_func(),
                    torch.nn.Linear(4*self._action_dim, 2*self._action_dim),
                )

        self.binary_left_emb = torch.nn.Sequential(
                    torch.nn.Linear(2*self._action_dim, 4*self._action_dim),
                    torch.nn.Dropout(p=2*dropout),
                    torch.nn.LayerNorm(4*self._action_dim),
                    self.activation_func(),
                    torch.nn.Linear(4*self._action_dim, 2*self._action_dim),
                )
        self.binary_right_emb = torch.nn.Sequential(
                    torch.nn.Linear(2*self._action_dim, 4*self._action_dim),
                    torch.nn.Dropout(p=2*dropout),
                    torch.nn.LayerNorm(4*self._action_dim),
                    self.activation_func(),
                    torch.nn.Linear(4*self._action_dim, 2*self._action_dim),
                )
        self.unary_left_emb = torch.nn.Sequential(
                    torch.nn.Linear(2*self._action_dim, 4*self._action_dim),
                    torch.nn.Dropout(p=2*dropout),
                    torch.nn.LayerNorm(4*self._action_dim),
                    self.activation_func(),
                    torch.nn.Linear(4*self._action_dim, 2*self._action_dim),
                )
        self.unary_right_emb = torch.nn.Sequential(
                    torch.nn.Linear(2*self._action_dim, 4*self._action_dim),
                    torch.nn.Dropout(p=2*dropout),
                    torch.nn.LayerNorm(4*self._action_dim),
                    self.activation_func(),
                    torch.nn.Linear(4*self._action_dim, 2*self._action_dim),
                )


        self.temperature_scale = torch.nn.Parameter(1*torch.ones(self.op_count))
        self.temperature_bias = torch.nn.Parameter(torch.zeros(self.op_count))

        self.cbr_utterance_pooler = BagOfEmbeddingsEncoder(embedding_dim=self._action_dim)
        self.cbr_utterance_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.cbr_type_embeddings = torch.nn.Embedding(len(self._op_names), self._action_dim)
        self.cbr_tree_type_embs = torch.nn.Embedding(self.op_count, self._action_dim)
        self.level_embeddings = torch.nn.Embedding(self._decoder_timesteps, self._action_dim)
         
        self.cbr_beam_enricher = PytorchTransformer(
                                                input_dim=self._action_dim,
                                                num_layers=2,
                                                feedforward_hidden_dim=4*self._action_dim,
                                                num_attention_heads=8,
                                                positional_encoding="embedding")
        self.cbr_tree_rep_transformer = PytorchTransformer(
                                                input_dim=self._action_dim,
                                                num_layers=1,
                                                feedforward_hidden_dim=4*self._action_dim,
                                                num_attention_heads=8,
                                                positional_encoding="embedding")
        self.cbr_utterance_augmenter = LxmertCrossAttentionLayer(
                                        hidden_size = self._action_dim,
                                        num_attention_heads = 8,
                                        attention_probs_dropout_prob = 0.1,
                                        ctx_dim = self._action_dim,
                                        hidden_dropout_prob = 0.1)
        self.cbr_enricher_pooler = BagOfEmbeddingsEncoder(embedding_dim=self._action_dim)
        self.beam_rep_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.enr_beam_rep_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.binary_left_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.binary_right_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.unary_left_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.unary_right_linear = torch.nn.Linear(self._action_dim, self._action_dim, bias=False)
        self.IGNORE_KEEP_FOR_CBR = False
        self.ENABLE_DEBUG_ASSERTIONS = True
        self.PRINT_DEBUG_MESSAGES = False 

    def set_flags(self):
        print("###########")
        print('Inside set_flags in models/smbop.py....')
        print("###########\n\n")
        print(self._misc_params)
        self.is_oracle = self._misc_params.get("is_oracle", False)
        self.ranking_ratio = self._misc_params.get("ranking_ratio", 0.7) # not used anywhwere ?
        self.unique_reranker = self._misc_params.get("unique_reranker", False) # not used anywhere ?
        self.cntx_reranker = self._misc_params.get("cntx_reranker", True) # not used much (not used practically)
        self.lin_after_cntx = self._misc_params.get("lin_after_cntx", False)
        self.utt_aug = self._misc_params.get("utt_aug", True)
        self.cntx_rep = self._misc_params.get("cntx_rep", False)
        self.add_residual_beam = self._misc_params.get("add_residual_beam", False) # not used anywhere ?
        self.add_residual_reranker = self._misc_params.get( # not used anywhere ?
            "add_residual_reranker", False
        )
        self.only_last_rerank = self._misc_params.get("only_last_rerank", False) # not used anywhere
        self.oldlstm = self._misc_params.get("oldlstm", False) # not used anywhere
        self.use_treelstm = self._misc_params.get("use_treelstm", False) # not used anywehere
        self.disentangle_cntx = self._misc_params.get("disentangle_cntx", True)
        self.cntx_beam = self._misc_params.get("cntx_beam", True) # whether to contextualize beam elements wrt each other via beam_encoder above?
        self.uniquify = self._misc_params.get("uniquify", True) # not used anywhere
        self.temperature = self._misc_params.get("temperature", 1.0)
        self.use_bce = self._misc_params["use_bce"] # not used anywhere
        self.value_pred = self._misc_params.get("value_pred", True)
        self.debug = self._misc_params.get("debug", False)

        self.reuse_cntx_reranker = self._misc_params.get("reuse_cntx_reranker", True) # not used anywhere
        self.should_rerank = self._misc_params.get("should_rerank", True) # not used anywhere

    def _create_type_tensor(self):
        rule_tensor = [
            [[0] * len(self._type_dict) for _ in range(len(self._type_dict))]
            for _ in range(len(self._type_dict))
        ] # op x op x op tensor
        if self.value_pred:
            RULES = node_util.RULES_values
        else:
            RULES = node_util.RULES_novalues

        rules = json.loads(RULES)
        for rule in rules:
            i, j_k = rule
            if len(j_k) == 0:
                continue
            elif len(j_k) == 2:
                j, k = j_k
            else:
                j, k = j_k[0], j_k[0]
            try:
                i, j, k = self._type_dict[i], self._type_dict[j], self._type_dict[k]
            except:
                continue
            rule_tensor[i][j][k] = 1
        self._rule_tensor = torch.tensor(rule_tensor)
        self._rule_tensor[self._type_dict["keep"]] = 1 #?
        self._rule_tensor_flat = self._rule_tensor.flatten()
        self._op_count = self._rule_tensor.size(0)

        self._term_ids = [
            self._type_dict[i]
            for i in [
                "Project",
                "Orderby_desc",
                "Limit",
                "Groupby",
                "intersect",
                "except",
                "union",
                "Orderby_asc",
            ]
        ]
        self._term_tensor = torch.tensor(
            [1 if i in self._term_ids else 0 for i in range(len(self._type_dict))]
        )

    def _create_action_dicts(self):
        unary_ops = [
            "keep",
            "min",
            "count",
            "max",
            "avg",
            "sum",
            "Subquery",
            "distinct",
            "literal",
        ]

        binary_ops = [
            "eq",
            "like",
            "nlike",
            "add",
            "sub",
            "nin",
            "lte",
            "lt",
            "neq",
            "in",
            "gte",
            "gt",
            "And",
            "Or",
            "except",
            "union",
            "intersect",
            "Product",
            "Val_list",
            "Orderby_desc",
            "Orderby_asc",
            "Project",
            "Selection",
            "Limit",
            "Groupby",
        ]
        self.binary_op_count = len(binary_ops)
        self.unary_op_count = len(unary_ops)
        self._op_names = [
            k for k in itertools.chain(binary_ops, unary_ops, ["nan", "Table", "Value"])
        ]
        self._type_dict = OrderedDict({k: i for i, k in enumerate(self._op_names)})
        self.keep_id = self._type_dict["keep"]
        self._ACTIONS = {k: 1 for k in unary_ops}
        self._ACTIONS.update({k: 2 for k in binary_ops})
        self._ACTIONS = OrderedDict(self._ACTIONS)
        self._frontier_size = sum(self._beam_size ** n for n in self._ACTIONS.values())
        self.hasher = None
        self.flag_move_to_gpu = True

    def move_to_gpu(self, device):
        if self.flag_move_to_gpu:
            self._term_tensor = self._term_tensor.to(device)
            self._rule_tensor_flat = self._rule_tensor_flat.to(device)
            self._rule_tensor = self._rule_tensor.to(device)
            self.flag_move_to_gpu = False
    """
    

#enc
the question concatenated with the schema

#db_id
the id of the database schema we want to execute the query against

#schema_hash (leaf_hash)
the hash of every schema string (applying dethash to every schema element) 

#schema_types
the type of every schema element (Value or Table), Value is either a Column or a literal.

#tree_obj
the AnyTree Node gold tree object after adding the hash attributes.

#gold_sql
the gold sql string.

#leaf_indices
makes it easier to pick the gold leaves during the oracle setup.

#entities
deprecated.

#orig_entities
used to reconstruct the tree for evaluation (this is added_values concatenated with the schema).

#is_gold_leaf
a boolean vector to tell if a given leaf is a gold leaf (i.e it corrosponds to a schema_hash that is in hash_gold_levelorder[0]).

#lengths
the length of the schema and the question, this is used to seperate them. 

#offsets
an array of size [batch_size, max_entity_token_length, 2] that contains the start and end indices for each schema token (and question, but that is inefficiet)
example:
given enc of [how,old,is,flights, flights, . ,start, flights, . ,end]
the output of batched_span_select given offsets would be:
[[how,pad,pad]
[[old,pad,pad]
..
[[flights,pad,pad]
[flights,.,start]
[flights,.,end]]

#relation
black box from ratsql

#depth
used to batch similar depth instances together. (see sorting keys in defaults.jsonnet)

#hash_gold_levelorder
An array of the gold hashes corrosponding to nodes in the gold tree
For example:
And  170816594
├── keep  -218759080
│   └── keep  -218759080
│       └── keep  -218759080
│           └── Value fictional_universe.type_of_fictional_setting -218759080
└── Join  -270677574
    ├── keep  55125446
    │   └── R  55125446
    │       └── Value fictional_universe.fictional_setting.setting_type 176689722
    └── Join  -149501965
        ├── keep  -94519500
        │   └── Value fictional_universe.fictional_setting.works_set_here -94519500
        └── literal  -26546860
            └── Value the atom cave raiders! 172249327
[[-218759080  176689722  -94519500  172249327]
 [-218759080   55125446  -94519500  -26546860]
 [-218759080   55125446 -149501965         -1]
 [-218759080 -270677574         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]]

#hash_gold_tree
root node hash

#span_hash
We apply dethash to every continuous span within the question, this results in a square matrix of size [batch_size, max_question_length, max_question_length].

#is_gold_span
a boolean vector to tell if a given span is a gold span (i.e it corrosponds to a span_hash that is in hash_gold_levelorder[0]).
    """
    def _flatten_cases_tensor(self,tensor):
        original_shape = list(tensor.shape)
        if len(original_shape) > 2:
            new_shape = [-1] + original_shape[2:]
        elif len(original_shape) == 2:
            new_shape = [-1]
        else:
            raise ValueError("tensor should have atleast two dimensions")
        new_tensor = tensor.reshape(new_shape)
        return new_tensor

    def _flatten_cases_list(self,ex_list):
        flattened_list = [item for sublist in ex_list for item in sublist]
        return flattened_list

    def forward(
        self,
        enc,
        db_id,
        leaf_hash, #schema_hash
        leaf_types, #schema_types
        tree_obj=None,
        gold_sql=None,
        leaf_indices=None,
        entities=None,
        orig_entities=None,
        is_gold_leaf=None,
        lengths=None,
        offsets=None,
        relation=None,
        depth=None,
        hash_gold_levelorder=None,
        hash_gold_tree=None,
        span_hash=None,
        is_gold_span=None,
    ):

        case_size = is_gold_span.shape[1]
        self.num_cases = case_size
        
        for key in enc["tokens"]:
            enc["tokens"][key] = self._flatten_cases_tensor(enc["tokens"][key])

        db_id = self._flatten_cases_list(db_id)
        leaf_hash = self._flatten_cases_tensor(leaf_hash)
        leaf_types = self._flatten_cases_tensor(leaf_types)
        tree_obj = self._flatten_cases_list(tree_obj)
        gold_sql = self._flatten_cases_list(gold_sql)
        leaf_indices = self._flatten_cases_tensor(leaf_indices)
        entities = self._flatten_cases_list(entities)
        orig_entities = self._flatten_cases_list(orig_entities)
        is_gold_leaf = self._flatten_cases_tensor(is_gold_leaf)
        lengths = self._flatten_cases_tensor(lengths)
        offsets = self._flatten_cases_tensor(offsets)
        relation = self._flatten_cases_tensor(relation)
        depth = self._flatten_cases_tensor(depth)
        hash_gold_levelorder = self._flatten_cases_tensor(hash_gold_levelorder)
        hash_gold_tree = self._flatten_cases_tensor(hash_gold_tree)
        span_hash = self._flatten_cases_tensor(span_hash)
        is_gold_span = self._flatten_cases_tensor(is_gold_span)

        batch_size = len(db_id)
        actual_batch_size = batch_size // case_size
        actual_batch_idx = torch.arange(actual_batch_size) * case_size
        boolean_batch_idx = torch.zeros(batch_size)
        boolean_batch_idx[actual_batch_idx]=1.0 
        list_actual_batch_idx = list(actual_batch_idx.numpy())
        actual_enc = {}
        actual_enc["tokens"] = {}
        for key in enc["tokens"]:
            actual_enc["tokens"][key] = enc["tokens"][key][actual_batch_idx] 

        total_start = time.time()
        outputs = {}
        beam_list = []
        item_list = []
        self._device = enc["tokens"]["token_ids"].device
        boolean_batch_idx = boolean_batch_idx.to(self._device)
        self.move_to_gpu(self._device)

        self.hasher = hashing.Hasher(self._device)
        (
            embedded_schema, # B x E x D ?
            schema_mask,
            embedded_utterance,
            utterance_mask,
        ) = self._encode_utt_schema(enc, offsets, relation, lengths)
        batch_size, utterance_length, _ = embedded_utterance.shape # B x T x D
        start = time.time()
        loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        pre_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        vector_loss = torch.tensor(
            [0] * batch_size, dtype=torch.float32, device=self._device
        )
        # tree_sizes_vector = torch.tensor(
        #     [0] * batch_size, dtype=torch.float32, device=self._device
        # )

        tree_sizes_vector = torch.tensor(
            [1] * batch_size, dtype=torch.float32, device=self._device
        ) # total tree nodes in loss
        if hash_gold_levelorder is not None:
            new_hash_gold_levelorder = hash_gold_levelorder.sort()[0].transpose(0, 1) #transpose to get levelorder
        if self.value_pred:
            span_scores, start_logits, end_logits = self.score_spans(
                embedded_utterance, utterance_mask
            ) # (B x T x T), (B x T), (B x T)
            span_mask = torch.isfinite(span_scores).bool()
            final_span_scores = span_scores.clone() #(B x T x T)
            delta = final_span_scores.shape[-1] - span_hash.shape[-1]
            span_hash = torch.nn.functional.pad(
                span_hash,
                pad=(0, delta, 0, delta),
                mode="constant",
                value=-1,
            )
            is_gold_span = torch.nn.functional.pad(
                is_gold_span,
                pad=(0, delta, 0, delta),
                mode="constant",
                value=0,
            )
            if self.training:
                batch_idx, start_idx, end_idx = is_gold_span.nonzero().t()
                final_span_scores[
                    batch_idx, start_idx, end_idx
                ] = allennlp.nn.util.max_value_of_dtype(final_span_scores.dtype) # to ensure gold spans in tok-k while training

                is_span_end = is_gold_span.sum(-2).float() # B x T
                is_span_start = is_gold_span.sum(-1).float() # B x T

                span_start_probs = allennlp.nn.util.masked_log_softmax( # B x T
                    start_logits, utterance_mask.bool(), dim=1
                )
                span_end_probs = allennlp.nn.util.masked_log_softmax( # B x T
                    end_logits, utterance_mask.bool(), dim=1
                )

            else:
                inv_boolean_batch_idx = (1 - boolean_batch_idx).reshape(-1,1,1).int()
                #assert torch.sum(inv_boolean_batch_idx) == batch_size - batch_size//case_size
                val_is_gold_span = is_gold_span * inv_boolean_batch_idx # do not consider gold for non-case examples during inference
                batch_idx, start_idx, end_idx = val_is_gold_span.nonzero().t()
                final_span_scores[
                    batch_idx, start_idx, end_idx
                ] = allennlp.nn.util.max_value_of_dtype(final_span_scores.dtype) # to ensure gold spans in tok-k while training
                #final_span_scores = final_span_scores #(B x T x T)            

            _, leaf_span_mask, best_spans = allennlp.nn.util.masked_topk(
                final_span_scores.view([batch_size, -1]),
                span_mask.view([batch_size, -1]),
                self._num_values,
            ) # _ , B x K, B x K 
            span_start_indices = best_spans // utterance_length # B x K
            span_end_indices = best_spans % utterance_length # B x K

            start_span_rep = allennlp.nn.util.batched_index_select(
                embedded_utterance.contiguous(), span_start_indices
            ) # B x K x D 
            end_span_rep = allennlp.nn.util.batched_index_select(
                embedded_utterance.contiguous(), span_end_indices
            ) # B x K x D
            span_rep = (end_span_rep + start_span_rep) / 2
            leaf_span_rep = span_rep
            leaf_span_hash = allennlp.nn.util.batched_index_select(
                span_hash.reshape([batch_size, -1, 1]), best_spans
            ).reshape([batch_size, -1]) # B x K (B x self._num_values)
            leaf_span_types = torch.where(
                leaf_span_mask, self._type_dict["Value"], self._type_dict["nan"]
            ).int() # B x K (B x self._num_values)

        leaf_schema_scores = self._rank_schema(embedded_schema) # B x E x 1?
        leaf_schema_scores = leaf_schema_scores / self.temperature # no temperature used for values?
        if is_gold_leaf is not None:
            is_gold_leaf = torch.nn.functional.pad(
                is_gold_leaf,
                pad=(0, leaf_schema_scores.size(-2) - is_gold_leaf.size(-1)),
                mode="constant",
                value=0,
            )

        if self.training:
            final_leaf_schema_scores = leaf_schema_scores.clone() # B x E x 1
            if not self.is_oracle:
                avg_leaf_schema_scores = allennlp.nn.util.masked_log_softmax( # B x E x 1
                    final_leaf_schema_scores,
                    schema_mask.unsqueeze(-1).bool(),
                    dim=1,
                )

            final_leaf_schema_scores = final_leaf_schema_scores.masked_fill( # B x E x 1 -- to keep gold schema values in top-k
                is_gold_leaf.bool().unsqueeze(-1),
                allennlp.nn.util.max_value_of_dtype(final_leaf_schema_scores.dtype),
            )
        else:
            val_is_gold_leaf = is_gold_leaf * (1 - boolean_batch_idx).reshape(-1,1).int() # do not use gold for non-cases during inference
            final_leaf_schema_scores = leaf_schema_scores.clone()
            final_leaf_schema_scores = final_leaf_schema_scores.masked_fill( # B x E x 1 -- to keep gold schema values in top-k
                val_is_gold_leaf.bool().unsqueeze(-1),
                allennlp.nn.util.max_value_of_dtype(final_leaf_schema_scores.dtype),
            )
            #final_leaf_schema_scores = leaf_schema_scores

        final_leaf_schema_scores = final_leaf_schema_scores.masked_fill( # B x E x 1
            ~schema_mask.bool().unsqueeze(-1),
            allennlp.nn.util.min_value_of_dtype(final_leaf_schema_scores.dtype),
        )

        min_k = torch.clamp(schema_mask.sum(-1), 0, self._n_schema_leafs) # B x 1
        _, leaf_schema_mask, top_beam_indices = allennlp.nn.util.masked_topk(
            final_leaf_schema_scores.squeeze(-1), mask=schema_mask.bool(), k=min_k
        ) # _, B x min_k.max(), B x min_k.max() 

        if self.is_oracle:

            leaf_indices = torch.nn.functional.pad(
                leaf_indices,
                pad=(0, self._n_schema_leafs - leaf_indices.size(-1)),
                mode="constant",
                value=-1,
            ) # B x K
            leaf_schema_mask = leaf_indices >= 0
            final_leaf_indices = torch.abs(leaf_indices)

        else:
            final_leaf_indices = top_beam_indices
            
        leaf_schema_rep = allennlp.nn.util.batched_index_select(
            embedded_schema.contiguous(), final_leaf_indices
        ) # B x K x D

        leaf_schema_hash = allennlp.nn.util.batched_index_select(
            leaf_hash.unsqueeze(-1), final_leaf_indices
        ).reshape([batch_size, -1]) # B x K 
        leaf_schema_types = (
            allennlp.nn.util.batched_index_select(
                leaf_types.unsqueeze(-1), final_leaf_indices
            )
            .reshape([batch_size, -1])
            .long()
        ) # B x K

        if self.value_pred:
            beam_rep = torch.cat([leaf_schema_rep, leaf_span_rep], dim=-2) # ? (B x K x D) & (B x K x D)
            beam_hash = torch.cat([leaf_schema_hash, leaf_span_hash], dim=-1) # (B x K) (B x K)
            beam_types = torch.cat([leaf_schema_types, leaf_span_types], dim=-1) # (B x K) (B x K)
            beam_mask = torch.cat([leaf_schema_mask, leaf_span_mask], dim=-1) # (B x K) (B x K)
            if self.training:
                item_list.append(
                    ra_postproc.ZeroItem(
                        beam_types,
                        final_leaf_indices,
                        span_start_indices,
                        span_end_indices,
                        orig_entities,
                        enc,
                        self.tokenizer,
                    )
                )
            else:
                item_list.append(
                    ra_postproc.ZeroItem(
                        beam_types[actual_batch_idx],
                        final_leaf_indices[actual_batch_idx],
                        span_start_indices[actual_batch_idx],
                        span_end_indices[actual_batch_idx],
                        [orig_entities[idx] for idx in list_actual_batch_idx],
                        actual_enc,
                        self.tokenizer,
                    )
                )
        else:
            beam_rep = leaf_schema_rep
            beam_hash = leaf_schema_hash
            beam_types = leaf_schema_types
            beam_mask = leaf_schema_mask
            if self.training:
                item_list.append(
                    ra_postproc.ZeroItem(
                        beam_types,
                        final_leaf_indices,
                        None,
                        None,
                        orig_entities,
                        enc,
                        self.tokenizer,
                    )
                )
            else:
                item_list.append(
                    ra_postproc.ZeroItem(
                        beam_types[actual_batch_idx],
                        final_leaf_indices[actual_batch_idx],
                        None,
                        None,
                        [orig_entities[idx] for idx in list_actual_batch_idx],
                        actual_enc,
                        self.tokenizer,
                    )
                )

        outputs["leaf_beam_hash"] = beam_hash
        outputs["hash_gold_levelorder"] = (batch_size*[None]) #?

        gold_reps_binary_left, gold_reps_binary_right, gold_masks_binary, \
        gold_types_binary, gold_types_binary_lchild, gold_types_binary_rchild, \
        gold_reps_unary_left, gold_reps_unary_right, gold_masks_unary, \
        gold_types_unary, gold_types_unary_lchild, gold_types_unary_rchild \
        = self.collect_gold_tree_reps(
                            beam_rep,
                            beam_hash,
                            beam_types,
                            beam_mask,
                            embedded_utterance, 
                            utterance_mask,
                            embedded_schema,
                            schema_mask,
                            final_span_scores,
                            span_mask,
                            is_gold_span,
                            span_hash,
                            final_leaf_schema_scores,
                            is_gold_leaf,
                            leaf_hash,
                            leaf_types,
                            new_hash_gold_levelorder,
                            hash_gold_levelorder,
                            tree_obj,
                            hash_gold_tree,
                            gold_sql)

        if not self.training:
            assert self.keep_id not in gold_types_binary_lchild[gold_masks_binary]
            assert self.keep_id not in gold_types_binary_rchild[gold_masks_binary]
            assert self.keep_id not in gold_types_unary_lchild[gold_masks_unary]
            assert self.keep_id not in gold_types_unary_rchild[gold_masks_unary]


        if self.PRINT_DEBUG_MESSAGES:
            G1 = gold_masks_binary.shape[1]
            G2 = gold_masks_unary.shape[1]
            global MAX_G1
            global MAX_G2
            MAX_G1 = max(G1,MAX_G1)
            MAX_G2 = max(G2, MAX_G2)

            print(gold_reps_binary_left.shape, gold_reps_unary_left.shape)
            print(f'\nMAX_G1: {MAX_G1}, MAX_G2: {MAX_G2}\n')
        

        l_child_idx = None
        r_child_idx = None
        old_beam_types = None
        cbr_beam_rep = beam_rep

        for decoding_step in range(self._decoder_timesteps):
            batch_size, seq_len, _ = beam_rep.shape
            if self.utt_aug:
                enriched_beam_rep = self._augment_with_utterance( # B x K x D
                    self._utterance_augmenter,
                    embedded_utterance,
                    utterance_mask,
                    beam_rep, # B x K x D
                    beam_mask,
                    ctx=self._beam_encoder,
                )
                cbr_enriched_beam_rep = self._augment_with_utterance( # B x K x D
                    self.cbr_utterance_augmenter,
                    embedded_utterance,
                    utterance_mask,
                    cbr_beam_rep, # B x K x D
                    beam_mask,
                    ctx=self._beam_encoder,
                )
            else:
                enriched_beam_rep = beam_rep
            if self.cntx_rep:
                beam_rep = enriched_beam_rep.contiguous()              

            # binary_ops_reps: [B x K**2 x D], unary_ops_reps: [B x K x D]
            frontier_scores, frontier_mask, binary_ops_reps, unary_ops_reps \
             = self.score_frontier( # B x (K**2 * binary_op_count + K*unary_op_count)
                enriched_beam_rep, beam_rep, beam_mask
            )

            frontier_scores = frontier_scores / self.temperature
            l_beam_idx, r_beam_idx = vec_utils.compute_beam_idx(  # B x (K**2 * binary_op_count + K*unary_op_count)
                batch_size,
                seq_len,
                self.binary_op_count,
                self.unary_op_count,
                device=self._device,
            )
            frontier_op_ids = vec_utils.compute_op_idx( # B x (K**2 * binary_op_count + K*unary_op_count)
                batch_size,
                seq_len,
                self.binary_op_count,
                self.unary_op_count,
                device=self._device,
            )

            frontier_hash = self.hash_frontier( # B x (K**2 * binary_op_count + K*unary_op_count)
                beam_hash, frontier_op_ids, l_beam_idx, r_beam_idx
            )
            valid_op_mask = self.typecheck_frontier( # B x (K**2 * binary_op_count + K*unary_op_count)
                beam_types, frontier_op_ids, l_beam_idx, r_beam_idx
            )
            frontier_mask = frontier_mask * valid_op_mask # B x (K**2 * binary_op_count + K*unary_op_count)

            unique_frontier_scores = frontier_scores # B x (K**2 * binary_op_count + K*unary_op_count)

            with torch.no_grad():
                is_levelorder_list = vec_utils.isin( #? # B x (K**2 * binary_op_count + K*unary_op_count)
                    new_hash_gold_levelorder[decoding_step + 1], frontier_hash
                )

            if self.training:

                avg_frontier_scores = allennlp.nn.util.masked_log_softmax( # B x (K**2 * binary_op_count + K*unary_op_count)
                    frontier_scores, frontier_mask.bool(), dim=1
                )
            else:
                unique_frontier_scores = unique_frontier_scores

            input_reps = self.combine_rich_beam_output(
                                        beam_rep=cbr_beam_rep, 
                                        enriched_beam_rep=cbr_enriched_beam_rep,
                                        beam_rep_left=None, 
                                        enriched_beam_rep_left=None,
                                        beam_rep_right=None, 
                                        enriched_beam_rep_right=None,
                                        root_node_types=beam_types,
                                        left_child_types=None,
                                        right_child_types=None, 
                                        embedded_utterance=embedded_utterance,
                                        utterance_mask=utterance_mask,
                                        decoding_step=decoding_step
                                        )

            input_reps_binary_left = self.binary_left_linear(input_reps)
            input_reps_binary_right = self.binary_right_linear(input_reps)
            input_reps_unary_left = self.unary_left_linear(input_reps)
            input_reps_unary_right = self.unary_right_linear(input_reps)


            input_child_types = beam_types 

            cbr_frontier_scores, \
            cbr_bce_positive, \
            cbr_bce_negative = self.get_cbr_frontier_scores(
                                    gold_reps_binary_left,
                                    gold_reps_binary_right,
                                    gold_masks_binary,
                                    gold_types_binary,
                                    gold_types_binary_lchild,
                                    gold_types_binary_rchild,
                                    gold_reps_unary_left,
                                    gold_reps_unary_right,
                                    gold_masks_unary,
                                    gold_types_unary,
                                    gold_types_unary_lchild,
                                    gold_types_unary_rchild,
                                    input_reps_binary_left,
                                    input_reps_binary_right,
                                    input_reps_unary_left,
                                    input_reps_unary_right,
                                    input_child_types
                                    )
            cbr_frontier_mask = frontier_mask            

            unique_cbr_frontier_scores = cbr_frontier_scores#.clone()

            combined_scores = self.combine_model_and_cbr_scores_into_probs(
                                unique_frontier_scores.detach(),
                                frontier_mask,
                                unique_cbr_frontier_scores,
                                cbr_frontier_mask,
                                frontier_op_ids
                                )

            if self.training:
                avg_cbr_frontier_scores = allennlp.nn.util.masked_log_softmax( # B x (K**2 * binary_op_count + K*unary_op_count)
                    cbr_frontier_scores, 
                    mask=frontier_mask.bool(), 
                    dim=1
                )
                float_is_levelorder_list = is_levelorder_list.float()
                loss_tensor = -avg_cbr_frontier_scores * float_is_levelorder_list
                loss_tensor[loss_tensor>1000]=0.0
                vector_loss += loss_tensor.squeeze().sum(-1)
                tree_sizes_vector += is_levelorder_list.bool().squeeze().sum(-1)
                
                
                avg_combined_scores = torch.log(combined_scores + 1e-20) # B x (K**2 * binary_op_count + K*unary_op_count)
                loss_tensor = -avg_combined_scores * float_is_levelorder_list
                vector_loss += loss_tensor.squeeze().sum(-1)
                
            else:
                unique_cbr_frontier_scores = unique_cbr_frontier_scores            

            if self.training:
                combined_scores = combined_scores.masked_fill( # B x (K**2 * binary_op_count + K*unary_op_count) 
                    is_levelorder_list.bool(),
                    allennlp.nn.util.max_value_of_dtype(combined_scores.dtype), # for gold ops to appear in topk
                )

            beam_scores, beam_mask, beam_idx = allennlp.nn.util.masked_topk( # B x 5K , B x 5K , B x 5K
                combined_scores, 
                mask=frontier_mask.bool(), k=self._beam_size
            )

            old_beam_types = beam_types.clone()

            beam_types = torch.gather(frontier_op_ids, -1, beam_idx) # B x 5K

            keep_indices = (beam_types == self.keep_id).nonzero().t().split(1) # no need to revise this after pruning as it's only used in _create_beam_rep            
            l_child_idx = torch.gather(l_beam_idx, -1, beam_idx) # B x 5K
            r_child_idx = torch.gather(r_beam_idx, -1, beam_idx) # B x 5K

            l_child_types = torch.gather(old_beam_types, -1, l_child_idx)
            r_child_types = torch.gather(old_beam_types, -1, r_child_idx)               
            

            child_types = allennlp.nn.util.batched_index_select( # B x 5K (used for dealing with keep ops)
                old_beam_types.unsqueeze(-1), r_child_idx # why not l_child_idx ?
            ).squeeze(-1)

            l_child_enriched_rep = allennlp.nn.util.batched_index_select(enriched_beam_rep, l_child_idx)
            r_child_enriched_rep = allennlp.nn.util.batched_index_select(enriched_beam_rep, r_child_idx)
            
            beam_rep, l_child_rep, r_child_rep = self._create_beam_rep(
                beam_rep, l_child_idx, r_child_idx, beam_types, keep_indices,
                return_child_reps=True, use_cbr=False
            ) # B x K x D

            cbr_beam_rep, cbr_l_child_rep, cbr_r_child_rep = self._create_beam_rep(
                cbr_beam_rep, l_child_idx, r_child_idx, beam_types, keep_indices,
                return_child_reps=True, use_cbr=True
            ) # B x K x D

            beam_hash = torch.gather(frontier_hash, -1, beam_idx) # B x K

            if decoding_step == 1 and self.debug:
                failed_list, node_list, failed_set = get_failed_set(
                    beam_hash,
                    decoding_step,
                    tree_obj,
                    batch_size,
                    hash_gold_levelorder,
                )
                if failed_set:
                    print("hi")
                    raise ValueError

            if self.training:
                item_list.append(
                ra_postproc.Item(beam_types, 
                                l_child_idx, 
                                r_child_idx, 
                                beam_mask)
                                )
            else:
                item_list.append(
                ra_postproc.Item(beam_types[actual_batch_idx], 
                                l_child_idx[actual_batch_idx], 
                                r_child_idx[actual_batch_idx], 
                                beam_mask[actual_batch_idx])
                                )
            beam_types = torch.where(
                beam_types == self.keep_id, child_types, beam_types
            )
            beam_list.append(
                [
                    beam_hash.clone(),
                    beam_mask.clone(),
                    beam_types.clone(),
                    beam_scores.clone(),
                ]
            )

        if not self.training:
            (
                beam_hash_list,
                beam_mask_list,
                beam_type_list,
                beam_scores_list,
            ) = zip(*beam_list)
            beam_mask_tensor = torch.cat(beam_mask_list, dim=1)
            beam_type_tensor = torch.cat(beam_type_list, dim=1)

            is_final_mask = (
                self._term_tensor[beam_type_tensor].bool().to(beam_mask_tensor.device)
            )
            beam_mask_tensor = beam_mask_tensor * is_final_mask
            beam_hash_tensor = torch.cat(beam_hash_list, dim=1)
            beam_scores_tensor = torch.cat(beam_scores_list, dim=1)
            beam_scores_tensor = beam_scores_tensor
            beam_scores_tensor = beam_scores_tensor.masked_fill(
                ~beam_mask_tensor.bool(),
                allennlp.nn.util.min_value_of_dtype(beam_scores_tensor.dtype),
            )

        if self.training:
            pre_loss = (vector_loss / tree_sizes_vector).mean()

            loss = pre_loss.squeeze()
            assert not bool(torch.isnan(loss))
            outputs["loss"] = loss
            self._compute_validation_outputs(
                outputs,
                hash_gold_tree,
                beam_hash,
            )
            return outputs
        else:
            end = time.time()
            outputs["leaf_beam_hash"] = outputs["leaf_beam_hash"][actual_batch_idx]
            outputs["hash_gold_levelorder"] = (actual_batch_size*[None]) 
            if tree_obj is not None:
                outputs["hash_gold_levelorder"] = [hash_gold_levelorder[actual_batch_idx]]+([None]*(actual_batch_size-1))
            self._compute_validation_outputs(
                outputs,
                hash_gold_tree[actual_batch_idx],
                beam_hash[actual_batch_idx],
                is_gold_leaf=is_gold_leaf[actual_batch_idx],
                top_beam_indices=top_beam_indices[actual_batch_idx],
                db_id=[db_id[idx] for idx in list_actual_batch_idx],
                beam_hash_tensor=beam_hash_tensor[actual_batch_idx],
                beam_scores_tensor=beam_scores_tensor[actual_batch_idx],
                gold_sql=[gold_sql[idx] for idx in list_actual_batch_idx],
                item_list=item_list, # cbr related modifications done in earlier parts of the file 
                inf_time=end - start,
                total_time=end - total_start,
            )
            return outputs

    @torch.no_grad()
    def score_spans(self, embedded_utterance, utterance_mask):
        logits = self._span_score_func(embedded_utterance) # B x T x 2
        logits = logits / self.temperature
        start_logits, end_logits = logits.split(1, dim=-1) # B x T x 1
        start_logits = start_logits.squeeze(-1) # B x T
        end_logits = end_logits.squeeze(-1) # B x T
        start_logits = vec_utils.replace_masked_values_with_big_negative_number(
            start_logits, utterance_mask
        ) # B x T
        end_logits = vec_utils.replace_masked_values_with_big_negative_number(
            end_logits, utterance_mask
        ) # B x T
        span_scores = vec_utils.get_span_scores(start_logits, end_logits) # B x T x T
        return span_scores, start_logits, end_logits

    #@torch.no_grad()
    def _create_beam_rep(
        self, beam_rep, l_child_idx, r_child_idx, beam_types, keep_indices, return_child_reps,
        use_cbr=False
    ):
        if use_cbr:
            type_embedding = self.cbr_tree_type_embs
            tree_rep_transformer = self.cbr_tree_rep_transformer
        else:
            type_embedding = self.type_embedding
            tree_rep_transformer = self._tree_rep_transformer

        l_child_rep = allennlp.nn.util.batched_index_select(beam_rep, l_child_idx)
        r_child_rep = allennlp.nn.util.batched_index_select(beam_rep, r_child_idx)
        beam_type_rep = type_embedding(beam_types)
        beam_rep = torch.stack([beam_type_rep, l_child_rep, r_child_rep], dim=-2)
        batch_size, beam_size, _, emb_size = beam_rep.shape
        beam_rep = beam_rep.reshape([-1, 3, self._action_dim])
        mask = torch.ones([beam_rep.size(0), 3], dtype=torch.bool, device=self._device)
        beam_rep = tree_rep_transformer(inputs=beam_rep, mask=mask)
        beam_rep = self._pooler(beam_rep).reshape([batch_size, beam_size, emb_size])

        beam_rep[keep_indices] = r_child_rep[keep_indices].type(beam_rep.dtype)
        if return_child_reps:
            return beam_rep, l_child_rep, r_child_rep
        else:
            return beam_rep

    def _compute_validation_outputs(
        self,
        outputs,
        hash_gold_tree,
        beam_hash,
        **kwargs,
    ):
        batch_size = beam_hash.size(0)
        final_beam_acc_list = []
        reranker_acc_list = []
        spider_acc_list = []
        leaf_acc_list = []
        sql_list = []
        top_k_sql_list = []
        tree_list = []
        beam_scores_el_list = []
        if hash_gold_tree is not None:
            for gs, fa in zip(hash_gold_tree, beam_hash.tolist()):
                acc = int(gs) in fa
                self._final_beam_acc(int(acc))
                final_beam_acc_list.append(bool(acc))

        if not self.training:

            if (
                kwargs["is_gold_leaf"] is not None
                and kwargs["top_beam_indices"] is not None
            ):
                for top_beam_indices_el, is_gold_leaf_el in zip(
                    kwargs["top_beam_indices"], kwargs["is_gold_leaf"]
                ):
                    is_gold_leaf_idx = is_gold_leaf_el.nonzero().squeeze().tolist()
                    if not isinstance(is_gold_leaf_idx, list):
                        is_gold_leaf_idx = [is_gold_leaf_idx]
                    leaf_acc = int(
                        all([x in top_beam_indices_el for x in is_gold_leaf_idx])
                    )
                    leaf_acc_list.append(leaf_acc)
                    self._leafs_acc(leaf_acc)

            # TODO: change this!! this causes bugs!
            for b in range(batch_size):
                beam_scores_el = kwargs["beam_scores_tensor"][b]
                beam_scores_el[
                    : -self._beam_size
                ] = allennlp.nn.util.min_value_of_dtype(beam_scores_el.dtype)
                beam_scores_el_list.append(beam_scores_el)
                top_idx = int(beam_scores_el.argmax())
                tree_copy = ""
                try:
                    items = kwargs["item_list"][: (top_idx // self._beam_size) + 2]

                    tree_res = ra_postproc.reconstruct_tree(
                        self._op_names,
                        self.binary_op_count,
                        b,
                        top_idx % self._beam_size,
                        items,
                        len(items) - 1,
                        self._n_schema_leafs,
                    )
                    tree_copy = deepcopy(tree_res)
                    sql = ra_postproc.ra_to_sql(tree_res)
                except:
                    print("Could not reconstruct SQL from RA tree")
                    sql = ""
                spider_acc = 0
                reranker_acc = 0

                top_k_sqls = self._get_top_k_sqls(beam_scores_el, kwargs["item_list"], b)

                outputs["inf_time"] = [kwargs["inf_time"]]+([None]*(batch_size-1))
                outputs["total_time"] = [kwargs["total_time"]] + \
                    ([None]*(batch_size-1))

                if hash_gold_tree is not None:
                    try:
                        reranker_acc = int(
                            kwargs["beam_hash_tensor"][b][top_idx]
                            == int(hash_gold_tree[b])
                        )

                        gold_sql = kwargs["gold_sql"][b]
                        db_id = kwargs["db_id"][b]
                        spider_acc = int(self._evaluate_func(gold_sql, sql, db_id))
                    except Exception as e:
                        print(f"EM evaluation failed {e}")

                reranker_acc_list.append(reranker_acc)
                self._reranker_acc(reranker_acc)
                self._spider_acc(spider_acc)
                sql_list.append(sql)
                top_k_sql_list.append(top_k_sqls)
                tree_list.append(tree_copy)
                spider_acc_list.append(spider_acc)
            outputs["beam_scores"] = beam_scores_el_list
            outputs["beam_encoding"] = [kwargs["item_list"]]+([None]*(batch_size-1))
            outputs["beam_hash"] = [kwargs["beam_hash_tensor"]]+([None]*(batch_size-1))
            # outputs["gold_hash"] = hash_gold_tree or ([None]*batch_size)
            if hash_gold_tree is not None:
                outputs["gold_hash"] = hash_gold_tree
            else:
                outputs["gold_hash"] = [hash_gold_tree] + ([None]*(batch_size-1))
            outputs["reranker_acc"] = reranker_acc_list
            outputs["spider_acc"] = spider_acc_list
            outputs["sql_list"] = sql_list
            outputs["top_k_sql_list"] = top_k_sql_list
            outputs["tree_list"] = tree_list
        outputs["final_beam_acc"] = final_beam_acc_list or ([None]*batch_size)
        outputs["leaf_acc"] = leaf_acc_list or ([None]*batch_size)

    def _get_top_k_sqls(self, beam_scores, item_list, batch_idx):
        #return []
        sql_list = []
        len_beam_scores = beam_scores.shape[0]
        lowest_allowed_idx = len_beam_scores - self._beam_size
        for i in (-beam_scores).argsort():
            if i < lowest_allowed_idx:
                continue
            try:
                items = item_list[: (i // self._beam_size) + 2]
                tree_res = ra_postproc.reconstruct_tree(
                    self._op_names, 
                    self.binary_op_count, 
                    batch_idx, 
                    i % self._beam_size, 
                    items, 
                    len(items)-1, 
                    self._n_schema_leafs)
                sql = ra_postproc.ra_to_sql(tree_res)
                sql_list.append(sql)
            except Exception as e:
                print(f'Error in getting top-k SQLs: {e}')
                continue
        assert len(sql_list) > 0
        return sql_list

    #@torch.no_grad()
    def _augment_with_utterance(
        self,
        utterance_augmenter,
        embedded_utterance,
        utterance_mask,
        beam_rep,
        beam_mask,
        ctx=None,
    ):
        assert ctx

        if self.disentangle_cntx:
            # first attend to input utterance
            # then contextualize the beam representations
            enriched_beam_rep = utterance_augmenter(
                beam_rep, embedded_utterance, ctx_att_mask=utterance_mask
            )[0]
            if self.cntx_beam:
                enriched_beam_rep = ctx(inputs=enriched_beam_rep, mask=beam_mask.bool())
        else:
            # directly contextualize beam w.r.t. to input utterance as well as itself
            # and return update beam representation
            encoder_input = torch.cat([embedded_utterance, beam_rep], dim=1)
            input_mask = torch.cat([utterance_mask.bool(), beam_mask.bool()], dim=-1)
            encoder_output = ctx(inputs=encoder_input, mask=input_mask)
            _, enriched_beam_rep = torch.split(
                encoder_output, [utterance_mask.size(-1), beam_mask.size(-1)], dim=1
            )

        return enriched_beam_rep

    def emb_q(self, enc):
        pad_dim = enc["tokens"]["mask"].size(-1)
        if pad_dim > 512: # hardcoding
            for key in enc["tokens"].keys():
                enc["tokens"][key] = enc["tokens"][key][:, :512] # hardcoding

            embedded_utterance_schema = self._question_embedder(enc)
        else:
            embedded_utterance_schema = self._question_embedder(enc)

        return embedded_utterance_schema

    @torch.no_grad()
    def _encode_utt_schema(self, enc, offsets, relation, lengths):
        embedded_utterance_schema = self.emb_q(enc)

        (
            embedded_utterance_schema,
            embedded_utterance_schema_mask,
        ) = vec_utils.batched_span_select(embedded_utterance_schema, offsets)
        embedded_utterance_schema = masked_mean(
            embedded_utterance_schema,
            embedded_utterance_schema_mask.unsqueeze(-1),
            dim=-2,
        )

        relation_mask = (relation >= 0).float()  # TODO: fixme
        torch.abs(relation, out=relation)
        embedded_utterance_schema = self._emb_to_action_dim(embedded_utterance_schema)
        enriched_utterance_schema = self._schema_encoder( # RAT Layers ?
            embedded_utterance_schema, relation.long(), relation_mask
        )

        utterance_schema, utterance_schema_mask = vec_utils.batched_span_select(
            enriched_utterance_schema, lengths
        )
        utterance, schema = torch.split(utterance_schema, 1, dim=1) #?
        utterance_mask, schema_mask = torch.split(utterance_schema_mask, 1, dim=1) #? dims?
        utterance_mask = torch.squeeze(utterance_mask, 1)
        schema_mask = torch.squeeze(schema_mask, 1)
        embedded_utterance = torch.squeeze(utterance, 1) #B x T x D
        schema = torch.squeeze(schema, 1)
        return schema, schema_mask, embedded_utterance, utterance_mask

    @torch.no_grad()
    def score_frontier(self, enriched_beam_rep, beam_rep, beam_mask):
        if self.cntx_rep: # default: False
            beam_rep = self._cntx_rep_linear(enriched_beam_rep)
        else:
            if self.utt_aug: # default True
                beam_rep = torch.cat([enriched_beam_rep, beam_rep], dim=-1)
                if self.lin_after_cntx: # default False
                    beam_rep = self.cntx_linear(beam_rep)
            else:
                beam_rep = self._nobeam_cntx_linear(beam_rep)

        batch_size, seq_len, emb_size = beam_rep.shape

        left = self.left_emb(beam_rep.reshape([batch_size, seq_len, 1, emb_size]))
        right = self.right_emb(beam_rep.reshape([batch_size, 1, seq_len, emb_size]))
        binary_ops_reps = self.after_add(left + right)
        binary_ops_reps = binary_ops_reps.reshape(-1, seq_len ** 2, self.d_frontier) # B x seq_len**2 x d_frontier
        unary_ops_reps = self._unary_frontier_embedder(beam_rep) # B x seq_len x d_frontier
        pre_frontier_rep = torch.cat([binary_ops_reps, unary_ops_reps], dim=1) # B x seq_len**2+seq_len x d_frontier
        pre_frontier_rep = self.pre_op_linear(pre_frontier_rep) # B x seq_len**2+seq_len x d_frontier  

        base_frontier_scores = self.op_linear(pre_frontier_rep) # B x seq_len**2+seq_len x binary_opcount+unary_opcount
        binary_frontier_scores, unary_frontier_scores = torch.split(
            base_frontier_scores, [seq_len ** 2, seq_len], dim=1
        ) # B x seq_len**2 x op_count , # B x seq_len x op_count
        binary_frontier_scores, _ = torch.split(
            binary_frontier_scores, [self.binary_op_count, self.unary_op_count], dim=2
        ) # B x seq_len**2 x binary_op_count
        _, unary_frontier_scores = torch.split(
            unary_frontier_scores, [self.binary_op_count, self.unary_op_count], dim=2
        ) # B x seq_len x unary_op_count
        frontier_scores = torch.cat(
            [
                binary_frontier_scores.reshape([batch_size, -1]),
                unary_frontier_scores.reshape([batch_size, -1]),
            ],
            dim=-1,
        ) # B x (seq_len**2 * binary_op_count + seq_len*unary_op_count)
        binary_mask = torch.einsum("bi,bj->bij", beam_mask, beam_mask)
        binary_mask = binary_mask.view([beam_mask.shape[0], -1]).unsqueeze(-1)
        binary_mask = binary_mask.expand(
            [batch_size, seq_len ** 2, self.binary_op_count]
        ).reshape(batch_size, -1)
        unary_mask = (
            beam_mask.clone()
            .unsqueeze(-1)
            .expand([batch_size, seq_len, self.unary_op_count])
            .reshape(batch_size, -1)
        )
        frontier_mask = torch.cat([binary_mask, unary_mask], dim=-1)
        binary_ops_reps, unary_ops_reps = torch.split(pre_frontier_rep,[seq_len**2, seq_len],dim=1)
        return frontier_scores, frontier_mask, binary_ops_reps, unary_ops_reps

    def hash_frontier(self, beam_hash, frontier_op_ids, l_beam_idx, r_beam_idx):
        r_hash = (
            allennlp.nn.util.batched_index_select(beam_hash.unsqueeze(-1), r_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        l_hash = (
            allennlp.nn.util.batched_index_select(beam_hash.unsqueeze(-1), l_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        tmp = l_hash.clone()
        frontier_hash = self.set_hash(
            frontier_op_ids.clone().reshape(-1), l_hash, r_hash
        ).long()
        frontier_hash = torch.where(
            frontier_op_ids.reshape(-1) == self.keep_id, tmp, frontier_hash
        )
        frontier_hash = frontier_hash.reshape(r_beam_idx.size())
        return frontier_hash

    def typecheck_frontier(self, beam_types, frontier_op_ids, l_beam_idx, r_beam_idx):
        batch_size, frontier_size = frontier_op_ids.shape

        r_types = (
            allennlp.nn.util.batched_index_select(beam_types.unsqueeze(-1), r_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        l_types = (
            allennlp.nn.util.batched_index_select(beam_types.unsqueeze(-1), l_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        indices_into = (
            self._op_count * self._op_count * frontier_op_ids.view(-1)
            + self._op_count * l_types
            + r_types
        )
        valid_ops = self._rule_tensor_flat[indices_into].reshape(
            [batch_size, frontier_size]
        )
        return valid_ops

    def set_hash(self, parent, a, b):
        a <<= 28
        b >>= 1
        a = a.add_(b)
        parent <<= 56
        a = a.add_(parent)
        a *= self.hasher.tensor2
        # TODO check lgu-lgm hashing instead of this:
        a = a.fmod_(self.hasher.tensor1)
        return a

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        out = {
            "final_beam_acc": self._final_beam_acc.get_metric(reset),
        }
        if not self.training:
            out["spider"] = self._spider_acc.get_metric(reset)
            out["reranker"] = self._reranker_acc.get_metric(reset)
            out["leafs_acc"] = self._leafs_acc.get_metric(reset)
            # out['self._spider_acc._count'] = self._spider_acc._count
        return out

    def get_cbr_frontier_scores(self,
                                gold_reps_binary_left,
                                gold_reps_binary_right,
                                gold_masks_binary,
                                gold_types_binary,
                                gold_types_binary_lchild,
                                gold_types_binary_rchild,
                                gold_reps_unary_left,
                                gold_reps_unary_right,
                                gold_masks_unary,
                                gold_types_unary,
                                gold_types_unary_lchild,
                                gold_types_unary_rchild,
                                input_reps_binary_left,
                                input_reps_binary_right,
                                input_reps_unary_left,
                                input_reps_unary_right,
                                input_child_types):
        '''
        gold_reps_binary: [bC x G x D]
        gold_reps_unary: [bC x G x D]
        gold_masks_binary: [bC x G]
        gold_masks_unary: [bC x G]
        gold_types: [bC x G]
        binary_ops_reps: [bC x K**2 x D]
        unary_ops_reps: [bC x K x D]
        '''

        DIST_TEMP = 1

        
        bC, G1, D1 = gold_reps_binary_left.shape
        bC_3, G2, D2 = gold_reps_unary_left.shape
        bC_1, K2, D3 = input_reps_binary_left.shape
        bC_2, K, D4 = input_reps_unary_left.shape
        assert bC==bC_1==bC_2==bC_3
        assert D1==D2==D3==D4
        assert K2 == K
        D = D1

        gold_reps_binary_left = gold_reps_binary_left.reshape(-1,self.num_cases*G1, D) # [b X CG x D]
        gold_reps_binary_right = gold_reps_binary_right.reshape(-1,self.num_cases*G1, D) # [b X CG x D]
        gold_reps_unary_left = gold_reps_unary_left.reshape(-1, self.num_cases*G2, D) # [b X CG x D]
        gold_reps_unary_right = gold_reps_unary_right.reshape(-1, self.num_cases*G2, D) # [b X CG x D]
        gold_masks_binary = gold_masks_binary.reshape(-1, self.num_cases, G1) # [b X C x G]
        gold_masks_unary = gold_masks_unary.reshape(-1, self.num_cases, G2) # [b X C x G]
        gold_types_binary = gold_types_binary.reshape(-1, self.num_cases, G1) # [b x C x G]
        gold_types_binary_lchild = gold_types_binary_lchild.reshape(-1, self.num_cases, G1)
        gold_types_binary_rchild = gold_types_binary_rchild.reshape(-1, self.num_cases, G1)
        gold_types_unary = gold_types_unary.reshape(-1, self.num_cases, G2) # [b x C x G]
        gold_types_unary_lchild = gold_types_unary_lchild.reshape(-1, self.num_cases, G2)
        gold_types_unary_rchild = gold_types_unary_rchild.reshape(-1, self.num_cases, G2)
        
        input_reps_binary_left = input_reps_binary_left.reshape(-1, self.num_cases*K2, D)
        input_reps_binary_right = input_reps_binary_right.reshape(-1, self.num_cases*K2, D)
        input_reps_unary_left = input_reps_unary_left.reshape(-1, self.num_cases*K, D)
        input_reps_unary_right = input_reps_unary_right.reshape(-1, self.num_cases*K, D)

        input_child_types = input_child_types.reshape(-1, self.num_cases, K) # [b x C x K]

        assert gold_reps_binary_left.shape[0] == bC//self.num_cases
        assert input_reps_unary_left.shape[0] == bC_2//self.num_cases

        if not self.training:
            gold_masks_binary[:,0,:] = False
            gold_masks_unary[:,0,:] = False

        self_mask = 1-torch.diag(torch.ones(self.num_cases, device=self._device)) # [C x C]
        self_mask = self_mask.bool()
        self_mask = self_mask.unsqueeze(1) # [C x 1 x C]
        self_mask = self_mask.unsqueeze(0).unsqueeze(-1) # [1 x C x 1 x C x 1]

        binary_types_match_lchild = gold_types_binary_lchild.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        binary_types_match_lchild = binary_types_match_lchild == input_child_types.unsqueeze(-1).unsqueeze(-1) # [b x C x K x C x G]
        binary_types_match_rchild = gold_types_binary_rchild.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        binary_types_match_rchild = binary_types_match_rchild == input_child_types.unsqueeze(-1).unsqueeze(-1) # [b x C x K x C x G]
        binary_types_match_mask = binary_types_match_lchild.unsqueeze(3) * binary_types_match_rchild.unsqueeze(2)
        binary_types_match_mask = binary_types_match_mask.reshape([-1, self.num_cases, K2**2, self.num_cases, G1])
        assert binary_types_match_mask.shape[0] == input_child_types.shape[0]





        gold_types_binary = gold_types_binary.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        #gold_types_binary = gold_types_binary.reshape(-1, 1, 1, self.num_cases*G1) # [b x 1 x 1 x CG]
        gold_types_binary = gold_types_binary.repeat(1, self.num_cases, K2**2, 1, 1) # [b x C x K2**2 x CG] 

        binary_ops_scores_left = -torch.cdist(input_reps_binary_left, 
                                            gold_reps_binary_left, p=2.0) # [b x C*K2 x C*G]
        binary_ops_scores_left = binary_ops_scores_left.reshape(-1, self.num_cases, K2, self.num_cases, G1) # [b x C x K2 x C x G]
        
        binary_ops_scores_right = -torch.cdist(input_reps_binary_right, 
                                            gold_reps_binary_right, p=2.0) # [b x C*K2 x C*G]
        binary_ops_scores_right = binary_ops_scores_right.reshape(-1, self.num_cases, K2, self.num_cases, G1) # [b x C x K2 x C x G]
       
        binary_ops_scores = binary_ops_scores_left.unsqueeze(3) + binary_ops_scores_right.unsqueeze(2) # [b x C x K2 x K2 x C x G]
        binary_ops_scores = binary_ops_scores/DIST_TEMP
        binary_ops_scores = binary_ops_scores.reshape(-1, self.num_cases, K2**2, self.num_cases, G1) # [b x C x K2**2 x C x G]
        binary_ops_mask = gold_masks_binary.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        binary_ops_mask = binary_ops_mask*self_mask # [b x C x 1 x C x G]
        # binary_ops_mask = binary_ops_mask * binary_types_match_mask

        binary_bce_logits = binary_ops_scores
        binary_bce_logits = (binary_bce_logits + 8)/2

        binary_bce_positive = self._bce_loss(binary_bce_logits, torch.ones_like(binary_bce_logits))
        binary_bce_positive = binary_bce_positive.masked_fill(~binary_ops_mask,0.0) 
        binary_bce_positive = binary_bce_positive.reshape(-1, self.num_cases, K2**2, self.num_cases*G1)
        binary_bce_negative = self._bce_loss(binary_bce_logits, torch.zeros_like(binary_bce_logits))
        binary_bce_negative = binary_bce_negative.masked_fill(~binary_ops_mask,0.0)
        binary_bce_negative = binary_bce_negative.reshape(-1, self.num_cases, K2**2, self.num_cases*G1)

        binary_ops_scores = binary_ops_scores.masked_fill(~binary_ops_mask,-float('inf')) # [b x C x K**2 x C x G]
        binary_ops_scores = binary_ops_scores.reshape(-1, self.num_cases, K2**2, self.num_cases*G1) # [b x C x K**2 x CG]

        gold_types_binary = gold_types_binary.reshape(-1, self.num_cases, K2**2, self.num_cases*G1) # [b x 1 x 1 x CG]
        

        # [b x C x K2**2 x O+U]
        binary_ops_scores = scatter_logsumexp(binary_ops_scores, gold_types_binary, 
                            dim=-1, dim_size=self.op_count)
        binary_ops_scores[binary_ops_scores<-5000.0] = -5000.0

        binary_bce_positive = scatter_mean(binary_bce_positive, gold_types_binary,
                                dim=-1, dim_size=self.op_count)
        binary_bce_negative = scatter_mean(binary_bce_negative, gold_types_binary,
                                dim=-1, dim_size=self.op_count)

        if self.ENABLE_DEBUG_ASSERTIONS:
            assert torch.all(binary_ops_scores[:,:,:,self.binary_op_count:]< -1000.0)
            assert torch.all(binary_bce_positive[:,:,:,self.binary_op_count:] == 0.0)
            assert torch.all(binary_bce_negative[:,:,:,self.binary_op_count:] == 0.0)
        
        binary_ops_scores = binary_ops_scores[:,:,:,0:self.binary_op_count] # [b x C x K**2 x O]
        binary_ops_scores = binary_ops_scores.reshape(-1,K2**2,self.binary_op_count) # [bC x K**2 x O]
        assert binary_ops_scores.shape[0] == bC \
        and binary_ops_scores.shape[1] == K2**2 \
        and binary_ops_scores.shape[2] == self.binary_op_count

        binary_ops_scores = binary_ops_scores.reshape(bC,-1)
        assert binary_ops_scores.shape[0] == bC \
        and binary_ops_scores.shape[1] == (K**2)*self.binary_op_count

        binary_bce_positive = binary_bce_positive[:,:,:,0:self.binary_op_count] # [b x C x K**2 x O]
        binary_bce_positive = binary_bce_positive.reshape(-1,K2**2,self.binary_op_count)
        binary_bce_positive = binary_bce_positive.reshape(bC,-1)
        binary_bce_negative = binary_bce_negative[:,:,:,0:self.binary_op_count] # [b x C x K**2 x O]
        binary_bce_negative = binary_bce_negative.reshape(-1,K2**2,self.binary_op_count)
        binary_bce_negative = binary_bce_negative.reshape(bC,-1)

        unary_types_match_lchild = gold_types_unary_lchild.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        unary_types_match_lchild = unary_types_match_lchild == input_child_types.unsqueeze(-1).unsqueeze(-1) # [b x C x K x C x G]
        unary_types_match_rchild = gold_types_unary_rchild.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        unary_types_match_rchild = unary_types_match_rchild == input_child_types.unsqueeze(-1).unsqueeze(-1) # [b x C x K x C x G]
        unary_types_match_mask = unary_types_match_lchild * unary_types_match_rchild
        assert unary_types_match_mask.shape[0] == input_child_types.shape[0]

        gold_types_unary = gold_types_unary.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        gold_types_unary = gold_types_unary.repeat(1, self.num_cases, K, 1, 1) # [b x C x K x CG]

        unary_ops_scores_left = -torch.cdist(input_reps_unary_left, gold_reps_unary_left, p=2.0) # [b x CK x CG]
        unary_ops_scores_right = -torch.cdist(input_reps_unary_right, gold_reps_unary_right, p=2.0) # [b x CK x CG]
        unary_ops_scores = unary_ops_scores_left + unary_ops_scores_right
        unary_ops_scores = unary_ops_scores.reshape(-1, self.num_cases, K, self.num_cases, G2)
        unary_ops_scores = unary_ops_scores/DIST_TEMP

        unary_ops_mask = gold_masks_unary.unsqueeze(1).unsqueeze(1) # [b x 1 x 1 x C x G]
        unary_ops_mask = unary_ops_mask*self_mask # [b x C x 1 x C x G]
        # unary_ops_mask = unary_ops_mask*unary_types_match_mask

        unary_bce_logits = unary_ops_scores
        unary_bce_logits = (unary_bce_logits + 8)/2

        unary_bce_positive = self._bce_loss(unary_bce_logits, torch.ones_like(unary_bce_logits))
        unary_bce_positive = unary_bce_positive.masked_fill(~unary_ops_mask,0.0) 
        unary_bce_positive = unary_bce_positive.reshape(-1, self.num_cases, K, self.num_cases*G2)
        unary_bce_negative = self._bce_loss(unary_bce_logits, torch.zeros_like(unary_bce_logits))
        unary_bce_negative = unary_bce_negative.masked_fill(~unary_ops_mask,0.0) 
        unary_bce_negative = unary_bce_negative.reshape(-1, self.num_cases, K, self.num_cases*G2)

        unary_ops_scores = unary_ops_scores.masked_fill(~unary_ops_mask,-float('inf')) # [b x C x K x C x G]
        unary_ops_scores = unary_ops_scores.reshape(-1, self.num_cases, K, self.num_cases*G2) # [b x C x K x CG]

        gold_types_unary = gold_types_unary.reshape(-1, self.num_cases, K, self.num_cases*G2)
        
        # [b x C x K x O+U]
        unary_ops_scores = scatter_logsumexp(unary_ops_scores, gold_types_unary, 
                            dim=-1, dim_size=self.op_count)
        unary_ops_scores[unary_ops_scores<-5000.0] = -5000.0


        unary_bce_positive = scatter_mean(unary_bce_positive, gold_types_unary,
                                        dim=-1, dim_size=self.op_count)
        unary_bce_negative = scatter_mean(unary_bce_negative, gold_types_unary,
                                dim=-1, dim_size=self.op_count)

        # IGNORE_KEEP_FOR_CBR (keep related assert)
        if self.IGNORE_KEEP_FOR_CBR and self.ENABLE_DEBUG_ASSERTIONS:
            assert torch.all(unary_ops_scores[:,:,:,:self.keep_id] < -1000.0)
            assert torch.all(unary_bce_positive[:,:,:,:self.keep_id] == 0.0)
            assert torch.all(unary_bce_negative[:,:,:,:self.keep_id] == 0.0)
        
        unary_ops_scores = unary_ops_scores[:,:,:,self.binary_op_count:] # [b x C x K x U]
        unary_ops_scores = unary_ops_scores.reshape(-1, K*self.unary_op_count) # [bC x KU]
        assert unary_ops_scores.shape[0] == bC and unary_ops_scores.shape[1] == K*self.unary_op_count

        unary_bce_positive = unary_bce_positive[:,:,:,self.binary_op_count:]
        unary_bce_positive = unary_bce_positive.reshape(-1, K*self.unary_op_count)
        unary_bce_negative = unary_bce_negative[:,:,:,self.binary_op_count:]
        unary_bce_negative = unary_bce_negative.reshape(-1, K*self.unary_op_count)

        cbr_frontier_scores = torch.cat([binary_ops_scores, unary_ops_scores],-1)
        assert cbr_frontier_scores.shape[0] == bC
        assert cbr_frontier_scores.shape[1] == (K**2)*self.binary_op_count+K*self.unary_op_count

        cbr_bce_positive = torch.cat([binary_bce_positive, unary_bce_positive], -1)
        cbr_bce_negative = torch.cat([binary_bce_negative, unary_bce_negative], -1)

        return cbr_frontier_scores, cbr_bce_positive, cbr_bce_negative

    def collect_gold_tree_reps(self,
                            beam_rep,
                            beam_hash,
                            beam_types,
                            beam_mask,
                            embedded_utterance, 
                            utterance_mask,
                            embedded_schema,
                            schema_mask,
                            final_span_scores,
                            span_mask,
                            is_gold_span,
                            span_hash,
                            final_leaf_schema_scores,
                            is_gold_leaf,
                            leaf_hash,
                            leaf_types,
                            new_hash_gold_levelorder,
                            hash_gold_levelorder,
                            tree_obj,
                            hash_gold_tree,
                            gold_sql):
        
        
        # =========== gold span extraction ================# 
        batch_size, utterance_length, _ = embedded_utterance.shape
        max_gold_spans = (is_gold_span.sum(-1).sum(-1)).max().int().item()
        max_gold_spans = max(max_gold_spans,1)
        _, leaf_span_mask, best_spans = allennlp.nn.util.masked_topk(
            final_span_scores.view([batch_size, -1]),
            mask=(span_mask.view([batch_size, -1]).bool()) * (is_gold_span.view([batch_size, -1]).bool()),
            k=max_gold_spans,
        ) # _ , B x K, B x K 
        span_start_indices = best_spans // utterance_length # B x K
        span_end_indices = best_spans % utterance_length # B x K

        start_span_rep = allennlp.nn.util.batched_index_select(
            embedded_utterance.contiguous(), span_start_indices
        ) # B x K x D 
        end_span_rep = allennlp.nn.util.batched_index_select(
            embedded_utterance.contiguous(), span_end_indices
        ) # B x K x D
        span_rep = (end_span_rep + start_span_rep) / 2
        leaf_span_rep = span_rep
        leaf_span_hash = allennlp.nn.util.batched_index_select(
            span_hash.reshape([batch_size, -1, 1]), best_spans
        ).reshape([batch_size, -1]) # B x K (B x self._num_values)
        leaf_span_types = torch.where(
            leaf_span_mask, self._type_dict["Value"], self._type_dict["nan"]
        ).int() # B x K (B x self._num_values)


        # ==================== gold leaf extraction =================#
        max_gold_leafs = is_gold_leaf.sum(-1).max().int().item()
        max_gold_leafs = max(max_gold_leafs,1)
        min_k = torch.clamp(schema_mask.sum(-1), 0, max_gold_leafs) # B x 1
        _, leaf_schema_mask, top_beam_indices = allennlp.nn.util.masked_topk(
            final_leaf_schema_scores.squeeze(-1), 
            mask=(schema_mask.bool() * is_gold_leaf.bool()), 
            k=min_k
        ) # _, B x min_k.max(), B x min_k.max() 

        if self.is_oracle:

            leaf_indices = torch.nn.functional.pad(
                leaf_indices,
                pad=(0, self._n_schema_leafs - leaf_indices.size(-1)),
                mode="constant",
                value=-1,
            ) # B x K
            leaf_schema_mask = leaf_indices >= 0
            final_leaf_indices = torch.abs(leaf_indices)

        else:
            final_leaf_indices = top_beam_indices
            
        leaf_schema_rep = allennlp.nn.util.batched_index_select(
            embedded_schema.contiguous(), final_leaf_indices
        ) # B x K x D

        leaf_schema_hash = allennlp.nn.util.batched_index_select(
            leaf_hash.unsqueeze(-1), final_leaf_indices
        ).reshape([batch_size, -1]) # B x K 
        leaf_schema_types = (
            allennlp.nn.util.batched_index_select(
                leaf_types.unsqueeze(-1), final_leaf_indices
            )
            .reshape([batch_size, -1])
            .long()
        ) # B x K

        if self.value_pred:
            beam_rep = torch.cat([leaf_schema_rep, leaf_span_rep], dim=-2) # ? (B x K x D) & (B x K x D)
            beam_hash = torch.cat([leaf_schema_hash, leaf_span_hash], dim=-1) # (B x K) (B x K)
            beam_types = torch.cat([leaf_schema_types, leaf_span_types], dim=-1) # (B x K) (B x K)
            beam_mask = torch.cat([leaf_schema_mask, leaf_span_mask], dim=-1) # (B x K) (B x K)
        else:
            beam_rep = leaf_schema_rep
            beam_hash = leaf_schema_hash
            beam_types = leaf_schema_types
            beam_mask = leaf_schema_mask

        # ===================================================#
        

        collected_reps_binary_left = []
        collected_reps_binary_right = []
        collected_reps_unary_left = []
        collected_reps_unary_right = []
        collected_masks_binary = []
        collected_masks_unary = []
        collected_types = []
        collected_types_lchild = []
        collected_types_rchild = []

        l_child_idx = None
        r_child_idx = None
        old_beam_types = None

        for decoding_step in range(self._decoder_timesteps):
            batch_size, seq_len, _ = beam_rep.shape
            
            if self.utt_aug:
                enriched_beam_rep = self._augment_with_utterance( # B x K x D
                    self.cbr_utterance_augmenter,
                    embedded_utterance,
                    utterance_mask,
                    beam_rep, # B x K x D
                    beam_mask,
                    ctx=self._beam_encoder,
                )
            else:
                enriched_beam_rep = beam_rep
            if self.cntx_rep:
                beam_rep = enriched_beam_rep.contiguous()

            with torch.no_grad():
                frontier_scores = self.get_pseudo_frontier_scores(batch_size, seq_len)
                frontier_mask = self.get_pseudo_frontier_mask(beam_mask)


            # binary_ops_reps: [B x K**2 x D], unary_ops_reps: [B x K x D]
            frontier_scores = frontier_scores / self.temperature
            l_beam_idx, r_beam_idx = vec_utils.compute_beam_idx(  # B x (K**2 * binary_op_count + K*unary_op_count)
                batch_size,
                seq_len,
                self.binary_op_count,
                self.unary_op_count,
                device=self._device,
            )
            frontier_op_ids = vec_utils.compute_op_idx( # B x (K**2 * binary_op_count + K*unary_op_count)
                batch_size,
                seq_len,
                self.binary_op_count,
                self.unary_op_count,
                device=self._device,
            )
            frontier_hash = self.hash_frontier( # B x (K**2 * binary_op_count + K*unary_op_count)
                beam_hash, frontier_op_ids, l_beam_idx, r_beam_idx
            )
            valid_op_mask = self.typecheck_frontier( # B x (K**2 * binary_op_count + K*unary_op_count)
                beam_types, frontier_op_ids, l_beam_idx, r_beam_idx
            )
            frontier_mask = frontier_mask * valid_op_mask # B x (K**2 * binary_op_count + K*unary_op_count)

            unique_frontier_scores = frontier_scores # B x (K**2 * binary_op_count + K*unary_op_count)

            #if self.training:
            if True:
                with torch.no_grad():
                    is_levelorder_list = vec_utils.isin( #? # B x (K**2 * binary_op_count + K*unary_op_count)
                        new_hash_gold_levelorder[decoding_step + 1], frontier_hash
                    )
                    gold_beam_size = is_levelorder_list.sum(-1).max().int().item()
                    if gold_beam_size == 0:
                        print('\n ===== \n WARNING: Encountered gold_beam_size == 0 .. Continuing! \n ====\n')
                        continue
                    #print(f'\n ===== \n gold_beam_size : {gold_beam_size} \n ====\n')

                unique_frontier_scores = unique_frontier_scores.masked_fill( # B x (K**2 * binary_op_count + K*unary_op_count) 
                    is_levelorder_list.bool(),
                    allennlp.nn.util.max_value_of_dtype(unique_frontier_scores.dtype), # for gold ops to appear in topk
                )

            gold_frontier_mask = frontier_mask.bool() *  is_levelorder_list.bool()
            # if is_levelorder_list[~(frontier_mask.bool())].sum() != 0:
            #     breakpoint()

            beam_scores, beam_mask, beam_idx = allennlp.nn.util.masked_topk( # B x K , B x K , B x K
                unique_frontier_scores, mask=gold_frontier_mask, k=gold_beam_size
            )

            #grand_beam_types = old_beam_types
            old_beam_types = beam_types.clone()

            beam_types = torch.gather(frontier_op_ids, -1, beam_idx) # B x K

            keep_indices = (beam_types == self.keep_id).nonzero().t().split(1)

            old_l_child_idx = l_child_idx
            old_r_child_idx = r_child_idx

            l_child_idx = torch.gather(l_beam_idx, -1, beam_idx) # B x K
            r_child_idx = torch.gather(r_beam_idx, -1, beam_idx) # B x K

            l_child_types = torch.gather(old_beam_types, -1, l_child_idx)
            r_child_types = torch.gather(old_beam_types, -1, r_child_idx)

            if self.ENABLE_DEBUG_ASSERTIONS:
                frontier_beam_mask = torch.gather(frontier_mask, -1, beam_idx).bool()
                assert torch.sum(beam_mask[~(frontier_beam_mask.bool())]) == 0
            
            child_types = allennlp.nn.util.batched_index_select(
                old_beam_types.unsqueeze(-1), r_child_idx # why not l_child_idx ?
            ).squeeze(-1)

            l_child_enriched_rep = allennlp.nn.util.batched_index_select(enriched_beam_rep, l_child_idx)
            r_child_enriched_rep = allennlp.nn.util.batched_index_select(enriched_beam_rep, r_child_idx)

            beam_rep, l_child_rep, r_child_rep = self._create_beam_rep(
                beam_rep, l_child_idx, r_child_idx, beam_types, keep_indices,
                return_child_reps=True, use_cbr=True
            ) # B x K x D

            beam_hash = torch.gather(frontier_hash, -1, beam_idx) # B x K
            if decoding_step == 1 and self.debug:
                failed_list, node_list, failed_set = get_failed_set(
                    beam_hash,
                    decoding_step,
                    tree_obj,
                    batch_size,
                    hash_gold_levelorder,
                )
                if failed_set:
                    print("hi")
                    raise ValueError

            binary_beam_mask = beam_types < self.binary_op_count # [B x K]
            
            # IGNORE_KEEP_FOR_CBR
            if self.IGNORE_KEEP_FOR_CBR:
                unary_beam_mask = beam_types > self.keep_id
            else:
                unary_beam_mask = beam_types >= self.keep_id
            
            assert self.binary_op_count == self.keep_id
            
            collected_types.append(beam_types)
            
            beam_types = torch.where(
                beam_types == self.keep_id, child_types, beam_types
            )

            collected_l_child_rep = self.combine_rich_beam_output(
                                                            beam_rep=l_child_rep, 
                                                            enriched_beam_rep=l_child_enriched_rep,
                                                            beam_rep_left=None, 
                                                            enriched_beam_rep_left=None,
                                                            beam_rep_right=None, 
                                                            enriched_beam_rep_right=None,
                                                            root_node_types=l_child_types,
                                                            left_child_types=None,
                                                            right_child_types=None, 
                                                            embedded_utterance=embedded_utterance,
                                                            utterance_mask=utterance_mask,
                                                            decoding_step=decoding_step
                                                            )

            collected_binary_rep_left = self.binary_left_linear(collected_l_child_rep)

            collected_r_child_rep = self.combine_rich_beam_output(
                                                            beam_rep=r_child_rep, 
                                                            enriched_beam_rep=r_child_enriched_rep,
                                                            beam_rep_left=None, 
                                                            enriched_beam_rep_left=None,
                                                            beam_rep_right=None, 
                                                            enriched_beam_rep_right=None,
                                                            root_node_types=r_child_types,
                                                            left_child_types=None,
                                                            right_child_types=None, 
                                                            embedded_utterance=embedded_utterance,
                                                            utterance_mask=utterance_mask,
                                                            decoding_step=decoding_step
                                                            )

            collected_binary_rep_right = self.binary_right_linear(collected_r_child_rep)



            collected_binary_mask = beam_mask * binary_beam_mask
            collected_unary_rep_left = self.unary_left_linear(collected_l_child_rep)
            collected_unary_rep_right = self.unary_right_linear(collected_r_child_rep)

            collected_unary_mask = beam_mask * unary_beam_mask

            collected_reps_binary_left.append(collected_binary_rep_left)
            collected_reps_binary_right.append(collected_binary_rep_right)
            collected_reps_unary_left.append(collected_unary_rep_left)
            collected_reps_unary_right.append(collected_unary_rep_right)
            collected_masks_binary.append(collected_binary_mask)
            collected_masks_unary.append(collected_unary_mask)

            collected_types_lchild.append(l_child_types)
            collected_types_rchild.append(r_child_types)

        collected_reps_binary_left = torch.cat(collected_reps_binary_left,1) # [B x G x D]
        collected_reps_binary_right = torch.cat(collected_reps_binary_right,1) # [B x G x D]
        collected_reps_unary_left = torch.cat(collected_reps_unary_left,1) # [B x G x D]
        collected_reps_unary_right = torch.cat(collected_reps_unary_right,1) # [B x G x D]
        collected_masks_binary = torch.cat(collected_masks_binary,1) # [B x G]
        collected_masks_unary = torch.cat(collected_masks_unary,1) # [B x G]
        collected_types = torch.cat(collected_types,1) # [B x G]
        collected_types_lchild = torch.cat(collected_types_lchild, 1)
        collected_types_rchild = torch.cat(collected_types_rchild, 1)

        ordered_reps_binary_left, ordered_reps_binary_right, ordered_masks_binary, \
        ordered_types_binary, ordered_lchild_types_binary, \
        ordered_rchild_types_binary = self.trim_non_gold_things(collected_reps_binary_left,
                                                    collected_reps_binary_right, 
                                                    collected_masks_binary,
                                                    collected_types,
                                                    collected_types_lchild,
                                                    collected_types_rchild)
        
        if self.ENABLE_DEBUG_ASSERTIONS:
            assert torch.all(ordered_types_binary[ordered_masks_binary]<self.keep_id)

        ordered_reps_unary_left, ordered_reps_unary_right, ordered_masks_unary, \
        ordered_types_unary, ordered_lchild_types_unary, \
        ordered_rchild_types_unary = self.trim_non_gold_things(collected_reps_unary_left,
                                                    collected_reps_unary_right, 
                                                    collected_masks_unary,
                                                    collected_types,
                                                    collected_types_lchild,
                                                    collected_types_rchild)

        # IGNORE_KEEP_FOR_CBR
        if self.ENABLE_DEBUG_ASSERTIONS:
            if self.IGNORE_KEEP_FOR_CBR:
                assert torch.all(ordered_types_unary[ordered_masks_unary]>self.keep_id)
            else:
                assert torch.all(ordered_types_unary[ordered_masks_unary]>=self.keep_id)

        if self.PRINT_DEBUG_MESSAGES:
            teacher_forcing_acc_list = []
            for gs, fa in zip(hash_gold_tree, beam_hash.tolist()):
                acc = int(gs) in fa
                teacher_forcing_acc_list.append(acc)
            print(f'Teacher forcing acc list: {teacher_forcing_acc_list}')



        return ordered_reps_binary_left, \
               ordered_reps_binary_right, \
               ordered_masks_binary, \
               ordered_types_binary, \
               ordered_lchild_types_binary, \
               ordered_rchild_types_binary, \
               ordered_reps_unary_left, \
               ordered_reps_unary_right, \
               ordered_masks_unary, \
               ordered_types_unary, \
               ordered_lchild_types_unary, \
               ordered_rchild_types_unary

    def trim_non_gold_things(self, reps_left, reps_right, masks, types, lchild_types, rchild_types):
        assert masks.dtype == torch.bool 
        masks = masks.to(torch.uint8)
        max_len = torch.sum(masks,-1).max()#.item()
        ordered_masks, ordered_indices = masks.sort(descending=True)
        ordered_masks = ordered_masks.bool()
        ordered_reps_left = torch.gather(reps_left, 1, ordered_indices.unsqueeze(-1).repeat(1,1,reps_left.shape[-1]))
        ordered_reps_right = torch.gather(reps_right, 1, ordered_indices.unsqueeze(-1).repeat(1,1,reps_right.shape[-1]))
        ordered_types = torch.gather(types, 1, ordered_indices)
        ordered_lchild_types = torch.gather(lchild_types, 1, ordered_indices)
        ordered_rchild_types = torch.gather(rchild_types, 1, ordered_indices)
        ordered_masks = ordered_masks[:,:max_len]
        ordered_reps_left = ordered_reps_left[:,:max_len,:]
        ordered_reps_right = ordered_reps_right[:,:max_len,:]
        ordered_types = ordered_types[:,:max_len]
        ordered_lchild_types = ordered_lchild_types[:,:max_len]
        ordered_rchild_types = ordered_rchild_types[:,:max_len]
        return ordered_reps_left, ordered_reps_right, ordered_masks, \
            ordered_types, ordered_lchild_types, ordered_rchild_types

    def get_pruned_binary_ops_reps(self,
                                unique_frontier_scores,
                                frontier_mask,
                                binary_ops_reps,
                                input_beam_size,
                                ):
        K = input_beam_size
        scores = unique_frontier_scores
        scores = scores.masked_fill(~frontier_mask,
                            allennlp.nn.util.min_value_of_dtype(scores.dtype)
                            )
        scores = scores[:, :(K**2)*self.binary_op_count]
        scores = scores.reshape(-1, K**2, self.binary_op_count)
        scores = scores.max(-1)[0] # [B x K**2]
        scores_mask = frontier_mask[:, :(K**2)*self.binary_op_count]
        scores_mask = scores_mask.reshape(-1, K**2, self.binary_op_count)
        scores_mask = scores_mask.sum(-1).bool() # [B x K**2]
        pruned_scores, pruned_mask, pruned_idx = allennlp.nn.util.masked_topk( # B x 5K , B x 5K , B x 5K
                            scores, 
                            mask=scores_mask, 
                            k=5*K
                        )
        pruned_binary_ops_reps = torch.gather(binary_ops_reps,1,
        pruned_idx.unsqueeze(-1).repeat(1,1,binary_ops_reps.shape[-1]))
        return pruned_binary_ops_reps, pruned_idx, pruned_mask

    def combine_model_and_cbr_scores_into_probs(
        self,
        frontier_scores,
        frontier_mask,
        cbr_frontier_scores,
        cbr_frontier_mask,
        frontier_op_ids):
        frontier_probs = allennlp.nn.util.masked_softmax(
                            frontier_scores,
                            frontier_mask.bool(),
                            memory_efficient=True,
                            dim=-1)
        cbr_frontier_probs = allennlp.nn.util.masked_softmax(
                            cbr_frontier_scores,
                            frontier_mask.bool(),
                            memory_efficient=True,
                            dim=-1)

        # IGNORE_KEEP_FOR_CBR
        if self.IGNORE_KEEP_FOR_CBR:
            keep_mask = (frontier_op_ids == self.keep_id) 
            total_keep_probs = (frontier_probs*(keep_mask.float())).sum(-1, keepdim=True)
            if self.ENABLE_DEBUG_ASSERTIONS:
                assert torch.all(cbr_frontier_probs[keep_mask] == 0)
            cbr_frontier_probs = (1-total_keep_probs)*cbr_frontier_probs
            cbr_frontier_probs[keep_mask] = frontier_probs[keep_mask]
        
        output_probs = (cbr_frontier_probs + frontier_probs)/2
        return output_probs

    def combine_model_and_cbr_scores_into_logits(
        self,
        frontier_scores,
        frontier_mask,
        cbr_frontier_scores,
        cbr_frontier_mask,
        frontier_op_ids):
        
        cbr_frontier_scores = (cbr_frontier_scores - cbr_frontier_scores.mean(-1, keepdims=True))\
                            / torch.sqrt(cbr_frontier_scores.var(-1, keepdims=True) + 1e-5)
        cbr_frontier_scores = self._cbr_norm_alpha*cbr_frontier_scores + self._cbr_norm_beta
        frontier_scores = (frontier_scores - frontier_scores.mean(-1, keepdims=True))\
                        / torch.sqrt(frontier_scores.var(-1, keepdims=True) + 1e-5)
        frontier_scores = self._frontier_norm_alpha*frontier_scores + self._frontier_norm_beta
        output_scores = frontier_scores + cbr_frontier_scores
        return output_scores

    def get_pseudo_frontier_scores(self, batch_size, seq_len):
        B = batch_size
        K = seq_len
        results = torch.randn([B, (K**2)*self.binary_op_count + K*self.unary_op_count], device=self._device)
        return results

    def get_pseudo_frontier_mask(self, beam_mask):
        batch_size, seq_len = beam_mask.shape
        binary_mask = torch.einsum("bi,bj->bij", beam_mask, beam_mask)
        binary_mask = binary_mask.view([beam_mask.shape[0], -1]).unsqueeze(-1)
        binary_mask = binary_mask.expand(
            [batch_size, seq_len ** 2, self.binary_op_count]
        ).reshape(batch_size, -1)
        unary_mask = (
            beam_mask.clone()
            .unsqueeze(-1)
            .expand([batch_size, seq_len, self.unary_op_count])
            .reshape(batch_size, -1)
        )
        frontier_mask = torch.cat([binary_mask, unary_mask], dim=-1)
        return frontier_mask

    def combine_enriched_and_beam_rep(self, beam_rep, enriched_beam_rep):
        #output = self.norm_beam_sum(beam_rep+enriched_beam_rep)
        #output = beam_rep + enriched_beam_rep
        if self.training:
            beam_rep = beam_rep#.detach()
            enriched_beam_rep = enriched_beam_rep#.detach()
        output = torch.cat([beam_rep, enriched_beam_rep],-1)
        ff_output = self.ff_combo(output)
        return self.norm_beam_sum(ff_output+output)

    def combine_rich_beam_output(self, 
                                beam_rep, 
                                enriched_beam_rep,
                                beam_rep_left, 
                                enriched_beam_rep_left,
                                beam_rep_right, 
                                enriched_beam_rep_right,
                                root_node_types,
                                left_child_types,
                                right_child_types, 
                                embedded_utterance,
                                utterance_mask,
                                decoding_step):
        """
        beam_rep : [B x K x D]
        enriched_beam_rep: [B x K x D]
        beam_rep_left: [B x K x D]
        beam_rep_right: [B x K x D]
        enriched_beam_rep_left: [B x K x D]
        enriched_beam_rep_right: [B x K x D]
        root_node_types: [B x K]
        left_child_types: [B x K]
        right_child_types: [B x K]
        embedded_utterance: [B x T x D]
        decoding_step: int
        """
        batch_size, seq_len, emb_size = beam_rep.shape
        pooled_utterance = (embedded_utterance * utterance_mask.unsqueeze(-1).float()).sum(1)
        pooled_utterance = self.cbr_utterance_linear(pooled_utterance)
        pooled_utterance = pooled_utterance.unsqueeze(1).repeat(1,seq_len,1) # [B x K x D]
        root_type_embs = self.cbr_type_embeddings(root_node_types) # [B x K x D]
        level_embs = decoding_step * torch.ones([batch_size, seq_len], dtype=torch.int, device=self._device)
        level_embs = self.level_embeddings(level_embs) # [B x K x D]

        joint_stack = [self.beam_rep_linear(beam_rep),
                      self.enr_beam_rep_linear(enriched_beam_rep),
                      root_type_embs,
                      level_embs,
                      pooled_utterance]

        stack_len = len(joint_stack)
        joint_embs = torch.stack(joint_stack, dim=-2)
        B, K, S, D = joint_embs.shape
        assert batch_size == B and K == seq_len and S == stack_len and D == emb_size
        joint_embs = joint_embs.reshape([-1, stack_len, emb_size])
        input_mask = torch.ones(B*K, stack_len, dtype=torch.bool, device=self._device)
        joint_rep = self.cbr_beam_enricher(inputs=joint_embs, mask=input_mask)
        joint_rep = self.cbr_enricher_pooler(joint_rep).reshape(B, K, D)
        # assert not torch.any(torch.isnan(joint_rep))
        return joint_rep

    def _get_schema_case_sim_scores(self, embedded_schema, schema_mask, 
                                    is_gold_leaf, case_size):
        batch_size, schema_size, emb_size = embedded_schema.shape
        actual_batch_size = batch_size//case_size
        embedded_schema = embedded_schema.reshape(actual_batch_size, case_size, schema_size, -1) # [b x C x E x D]
        assert embedded_schema.shape[-1] == emb_size
        schema_mask = schema_mask.reshape(actual_batch_size, case_size, schema_size) # [b x C x E]
        copy_schema_mask = schema_mask.unsqueeze(1)
        schema_mask = schema_mask.unsqueeze(2) * copy_schema_mask # [b x C x C x E]
        copy_embedded_schema = embedded_schema.unsqueeze(1)
        embedded_schema = embedded_schema.unsqueeze(2)        
        sim_leaf_schema_scores = -torch.linalg.norm(embedded_schema - copy_embedded_schema, dim=-1)
        diag_mask = 1-torch.diag(torch.ones(case_size)) # [C x C]
        diag_mask = diag_mask.reshape(1, case_size, case_size, 1)
        diag_mask = diag_mask.to(self._device)
        is_gold_leaf = is_gold_leaf.reshape(actual_batch_size, case_size, schema_size) # [b x C x E]
        is_gold_leaf = is_gold_leaf.unsqueeze(1) # [b x 1 x C x E]
        score_mask = diag_mask * schema_mask * is_gold_leaf
        assert torch.sum(score_mask) > 0
        min_sim = -1000
        sim_leaf_schema_scores = (sim_leaf_schema_scores*score_mask) + min_sim*(1-score_mask)
        sim_leaf_schema_scores = allennlp.nn.util.logsumexp(sim_leaf_schema_scores, 2) # [b x C x E]
        sim_leaf_schema_scores = sim_leaf_schema_scores.reshape(-1, schema_size) # [B x E]
        assert sim_leaf_schema_scores.shape[0] == batch_size
        return sim_leaf_schema_scores.unsqueeze(-1) # [B x E x 1]

    def _get_schema_case_sim_scores_cosine(self, embedded_schema, schema_mask, 
                                    is_gold_leaf, case_size):
        batch_size, schema_size, emb_size = embedded_schema.shape
        actual_batch_size = batch_size//case_size
        embedded_schema = embedded_schema.reshape(actual_batch_size, case_size, schema_size, -1) # [b x C x E x D]
        assert embedded_schema.shape[-1] == emb_size
        schema_mask = schema_mask.reshape(actual_batch_size, case_size, schema_size) # [b x C x E]
        copy_schema_mask = schema_mask.unsqueeze(1)
        schema_mask = schema_mask.unsqueeze(2) * copy_schema_mask # [b x C x C x E]
        copy_embedded_schema = embedded_schema.unsqueeze(1)
        embedded_schema = embedded_schema.unsqueeze(2)
        sim_leaf_schema_scores = torch.nn.functional.cosine_similarity(embedded_schema, copy_embedded_schema,-1) # [b x C x C x E]
        diag_mask = 1-torch.diag(torch.ones(case_size)) # [C x C]
        diag_mask = diag_mask.reshape(1, case_size, case_size, 1)
        diag_mask = diag_mask.to(self._device)
        is_gold_leaf = is_gold_leaf.reshape(actual_batch_size, case_size, schema_size) # [b x C x E]
        is_gold_leaf = is_gold_leaf.unsqueeze(1) # [b x 1 x C x E]
        score_mask = diag_mask * schema_mask * is_gold_leaf
        assert torch.sum(score_mask) > 0
        min_sim = -1
        sim_leaf_schema_scores = (sim_leaf_schema_scores*score_mask) + min_sim*(1-score_mask)
        sim_leaf_schema_scores = allennlp.nn.util.logsumexp(100*sim_leaf_schema_scores, 2)/100 # [b x C x E]
        sim_leaf_schema_scores = sim_leaf_schema_scores.reshape(-1, schema_size) # [B x E]
        sim_leaf_schema_scores = torch.nn.functional.relu(sim_leaf_schema_scores)
        assert sim_leaf_schema_scores.shape[0] == batch_size
        return sim_leaf_schema_scores.unsqueeze(-1) # [B x E x 1]


def get_failed_set(
    beam_hash, decoding_step, tree_obj, batch_size, hash_gold_levelorder
):
    failed_set = []
    failed_list = []
    node_list = []
    for b in range(batch_size):
        node_list.append(node_util.print_tree(tree_obj[b]))
        node_dict = {node.hash: node for node in PostOrderIter(tree_obj[b])}
        batch_set = (
            set(hash_gold_levelorder[b][decoding_step + 1].tolist())
            - set(beam_hash[b].tolist())
        ) - {-1}
        failed_list.append([node_dict[set_el] for set_el in batch_set])
        failed_set.extend([node_dict[set_el] for set_el in batch_set])
    return failed_list, node_list, failed_set