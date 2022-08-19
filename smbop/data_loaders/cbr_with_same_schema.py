from collections import deque
import logging
from multiprocessing.process import BaseProcess
import random
import traceback
from typing import List, Iterator, Optional, Iterable, Union, TypeVar

from overrides import overrides
import torch
import torch.multiprocessing as mp

from allennlp.common.util import lazy_groups_of, shuffle_iterable
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo, DatasetReaderInput
from allennlp.data.fields import TextField
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util
from allennlp.data.data_loaders import MultiProcessDataLoader

import pickle
import numpy as np
from allennlp.data import  Instance
from allennlp.data.fields import  ListField, MetadataField, ArrayField
logger = logging.getLogger(__name__)

_T = TypeVar("_T")

@DataLoader.register("cbr_with_same_schema")
class CBRSameSchemaDataLoader(MultiProcessDataLoader):
    def __init__(
        self,
        reader: DatasetReader,
        data_path: DatasetReaderInput,
        *,
        batch_size: int = None,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: BatchSampler = None,
        batches_per_epoch: int = None,
        num_workers: int = 0,
        max_instances_in_memory: int = None,
        start_method: str = "fork",
        cuda_device: Optional[Union[int, str, torch.device]] = None,
        quiet: bool = False,
    ) -> None:
        super().__init__(
            reader,
            data_path,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            batches_per_epoch=batches_per_epoch,
            num_workers=num_workers,
            max_instances_in_memory=max_instances_in_memory,
            start_method=start_method,
            cuda_device=cuda_device,
            quiet=quiet)

        ########################### added by ali/aa
        self.same_db_nbrs = self.reader.neighbours
        self.is_training = self.reader.is_training
        self.all_instances=self.reader.all_instances
        if self.same_db_nbrs is None:
            self.same_db_nbrs = [[j for j in range(len(self.all_instances)) if j!=i] 
                                 for i in range(len(self.all_instances))]
        ########################### added by ali/aa

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._maybe_tqdm(self._instances):
                instance.index_fields(vocab)

    def _instances_to_batches(
        self, instance_iterator: Iterable[Instance], move_to_device
    ) -> Iterator[TensorDict]:
        instance_iterator = (self._index_instance(instance) for instance in instance_iterator)
        if move_to_device and self.cuda_device is not None:
            tensorize = lambda batch: nn_util.move_to_device(  # noqa: E731
                self.collate_fn(batch), self.cuda_device
            )
        else:
            tensorize = self.collate_fn

        if self.batch_sampler is not None:
            instance_chunks: Iterable[List[Instance]]

            if self.max_instances_in_memory is not None:
                instance_chunks = lazy_groups_of(instance_iterator, self.max_instances_in_memory)
            else:
                instance_chunks = [list(instance_iterator)]

            for instances in instance_chunks:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
                ########################### added by ali/aa
                for batch in batches:
                    new_batch = self._modify_batch(batch)
                    yield tensorize(new_batch)
                ###########################
                    
        else:
            # Safe to assume this is not `None` when `self.batch_sampler` is `None`.
            assert self.batch_size is not None

            if self.shuffle:
                if self.max_instances_in_memory is not None:
                    instance_iterator = shuffle_iterable(
                        instance_iterator,
                        self.max_instances_in_memory,
                    )
                else:
                    # At this point we've already loaded the instances in memory and indexed them,
                    # so this won't take long.
                    instance_iterator = list(instance_iterator)
                    random.shuffle(instance_iterator)

            for batch in lazy_groups_of(instance_iterator, self.batch_size):
                if self.drop_last and len(batch) < self.batch_size:
                    break
                new_batch = self._modify_batch(batch)
                yield tensorize(new_batch)

    def _modify_batch(self, batch):
        MAX_NBRS = 20
        NUM_NBRS = 5
        new_batch=[]
        for inst in batch:
            iid=inst['inst_id'].metadata
            if len(self.same_db_nbrs[iid]) == 0:
                print('\n=============\n')
                print(f'WARNING: Instance {iid} has no same db nbr')
                print('\n=============\n')
                continue
            if self.is_training:
                same_db_ids = np.random.choice(self.same_db_nbrs[iid],31) # hardcoding
                #same_db_ids = np.random.choice(self.same_db_nbrs[iid][0:MAX_NBRS],NUM_NBRS) # hardcoding
                #same_db_ids = np.random.choice(len(self.all_instances), 5)
            else:
                same_db_ids = np.random.choice(self.same_db_nbrs[iid],31) # hardcoding
                #same_db_ids = self.same_db_nbrs[iid][0:NUM_NBRS] # hardcoding
                #same_db_ids = np.random.choice(len(self.all_instances), 10)
                
            for idx in same_db_ids:
                assert self.all_instances[idx]['db_id'].metadata == inst['db_id'].metadata

            nbr_instances = [self._index_instance(self.all_instances[nbr]) for nbr in same_db_ids]
            ins_fields = {}
            field_names = inst.fields.keys()
            for field_type in field_names:
                if field_type == 'inst_id':
                    continue
                ex_item = inst[field_type]
                nn_items = [item[field_type] for item in nbr_instances]
                all_items = [ex_item] + nn_items
                list_field = ListField(all_items)
                ins_fields[field_type] = list_field
            new_batch.append(Instance(ins_fields))
        return new_batch
