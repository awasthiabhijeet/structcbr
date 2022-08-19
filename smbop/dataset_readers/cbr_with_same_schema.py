import pickle

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField, ArrayField
from typing import Dict
from overrides import overrides

from smbop.dataset_readers import spider
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader

@DatasetReader.register("cbr_with_same_schema")
class CBRSameSchemaDatasetReader(SmbopSpiderDatasetReader):
    def __init__(
        self,
        lazy: bool = True,
        question_token_indexers: Dict[str, TokenIndexer] = None,
        keep_if_unparsable: bool = True,
        tables_file: str = None,
        dataset_path: str = "dataset/database",
        cache_directory: str = "cache/train",
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        max_instances=10000000,
        decoder_timesteps=9,
        limit_instances=-1,
        value_pred=True,
        use_longdb=True,
        is_training=True,
        neighbour_file: str = None,
    ):
        super().__init__(
            lazy,
            question_token_indexers,
            keep_if_unparsable,
            tables_file,
            dataset_path,
            cache_directory,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            max_instances,
            decoder_timesteps,
            limit_instances,
            value_pred,
            use_longdb,
        )
        self.is_training = is_training
        if neighbour_file is not None:
            self.neighbours = pickle.load(open(neighbour_file,'rb'))
            print(f'\n**** loaded neighbours from {neighbour_file} ***\n')
        else:
            self.neighbours = None

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".pkl"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        print('Reading: ',file_path)
        cnt = 0
        cache_buffer = []
        cont_flag = True
        if cont_flag:
            with open(file_path, "rb") as data_file:
                pkl_obj = pickle.load(data_file)
                [item.add_field('inst_id',MetadataField(i)) for i,item in enumerate(pkl_obj)]
                self.all_instances = pkl_obj
                for total_cnt, ex in enumerate(pkl_obj):
                    if cnt >= self._max_instances:
                        break
                    yield ex
                    cnt+=1

    def apply_token_indexers(self, instance: Instance) -> None:
        if isinstance(instance.fields["enc"], ListField):
            for text_field in instance.fields["enc"].field_list:
                text_field.token_indexers = self._utterance_token_indexers
        else:
            instance.fields["enc"].token_indexers = self._utterance_token_indexers





