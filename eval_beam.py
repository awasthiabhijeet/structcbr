import argparse
import torch

from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
import json
from allennlp.common import Params

from smbop.dataset_readers.cbr_with_same_schema import CBRSameSchemaDatasetReader
from smbop.dataset_readers.pickle_reader import PickleReader
from smbop.dataset_readers.cbr_concat import CBRConcat
from smbop.dataset_readers.cbr_concat_roberta import CBRConcatRoberta

from smbop.data_loaders.cbr_with_same_schema import CBRSameSchemaDataLoader
from smbop.data_loaders.cbr_concat import CBRConcat
from smbop.data_loaders.cbr_concat_roberta import CBRConcatRoberta

from smbop.models.smbop import SmbopParser
from smbop.models.tx_cbr_improved_entire_frontier import TXCBRImprovedEntireFrontier
from smbop.models.cbr_concat import CBRConcat
from smbop.models.cbr_concat_roberta import CBRConcatRoberta


from smbop.modules.relation_transformer import RelationTransformer
from smbop.modules.lxmert import LxmertCrossAttentionLayer

from smbop.eval_final.evaluation import eval_exec_match, \
rebuild_sql_val, rebuild_sql_col, build_valid_col_units, build_foreign_key_map_from_json, evaluate_single

try:
    from smbop.eval_final.process_sql import (
        tokenize,
        get_schema,
        get_tables_with_alias,
        Schema,
        get_sql,
    )
except:
    from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql

import itertools
import smbop.utils.node_util as node_util
import numpy as np
import numpy as np
import json
import tqdm
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import json
import pickle
import os
import shutil
import tarfile
from pathlib import Path
from allennlp.data.fields import ListField
from collections import defaultdict

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


def sanitize(query):
    query = query.replace(")", " ) ")
    query = query.replace("(", " ( ")
    query = ' '.join(query.split())
    query = query.replace('> =', '>=')
    query = query.replace('< =', '<=')
    query = query.replace('! =', '!=')

    query = query.replace('"', "'")
    if query.endswith(";"):
        query = query[:-1]
    for i in [1, 2, 3, 4, 5]:
        query = query.replace(f"t{i}", f"T{i}")
    for agg in ["count", "min", "max", "sum", "avg"]:
        query = query.replace(f"{agg} (", f"{agg}(")
    for agg in ["COUNT", "MIN", "MAX", "SUM", "AVG"]:
        query = query.replace(f"{agg} (", f"{agg}(")
    for agg in ["Count", "Min", "Max", "Sum", "Avg"]:
        query = query.replace(f"{agg} (", f"{agg}(")
    return query

@torch.no_grad()
def get_reranked_sqls(model, tokenizer, top_k_sqls, questions):
    print('\nReranking SQLs...\n')
    output_sqls = [] 
    for i,q in enumerate(questions):
        #breakpoint()
        sqls = top_k_sqls[i][0:5]
        sqls = [sanitize(el) for el in sqls]
        ques = [q] * len(sqls)    
        inputs = tokenizer(text=ques, text_pair=sqls, return_tensors='pt', padding=True)
        output = model(**inputs)
        score = output['logits'].reshape(-1)
        score = score.cpu().numpy()
        argmax = np.argmax(score)
        output_sqls.append(top_k_sqls[i][argmax])
        # print(f'scores: {score}')
        # print(f'max_score: {score[argmax]}, max_score_idx: {argmax}')
        # print()
    return output_sqls

def create_archive(archive_dir):
    filename = os.path.join(archive_dir,'model.tar.gz')
    tmp_dir = os.path.join(archive_dir,'model')
    os.makedirs(tmp_dir)
    best_weights = os.path.join(archive_dir,'best.th')
    shutil.copyfile(best_weights, os.path.join(tmp_dir,'weights.th'))
    config_file = os.path.join(archive_dir, 'config.json')
    shutil.copyfile(config_file, os.path.join(tmp_dir,'config.json'))
    vocab_file = os.path.join(archive_dir, 'vocabulary/non_padded_namespaces.txt')
    os.makedirs(os.path.join(tmp_dir,'vocabulary'))
    shutil.copyfile(vocab_file, os.path.join(tmp_dir,'vocabulary/non_padded_namespaces.txt'))
    with tarfile.open(filename, "w:gz") as tar:
        tar.add(tmp_dir, arcname=os.path.basename(tmp_dir))
    shutil.rmtree(tmp_dir)
    return filename

def update_instance_with_cases(ex, cases, idx):
    ins_fields={}
    field_names = ex.fields.keys()    
    #case_indices = np.random.choice(len(cases),3)
    #cases = [cases[i] for i in case_indices]
    #cases = [ex]
    for field_type in field_names:
        ex_item = ex[field_type]
        case_items = [item[field_type] for item in cases]
        #print('\n=============\n', len(cases),'\n=============\n')
        #case_items = [cases[idx][field_type]]
        if field_type == 'gold_sql':
            pass
            #print('case: ',case_items[0].metadata)
        all_items = [ex_item] + case_items
        list_field = ListField(all_items)
        ins_fields[field_type] = list_field
    instance = Instance(ins_fields)
    return instance

def add_cases_to_instance(instance):
    # used for gold retrieved pickles
    ins_fields={}
    field_names = instance.fields.keys()
    cases = instance["cases"].metadata
    cases = [item[0] for item in cases]
    ex = instance
    for field_type in field_names:
        if field_type == "cases":
            continue
        ex_item = ex[field_type]
        case_items = [item[field_type] for item in cases]
        all_items = [ex_item] + case_items
        list_field = ListField(all_items)
        ins_fields[field_type] = list_field
    instance = Instance(ins_fields)
    return instance

def eval_exec_single(g_str, p_str, db_id, db_dir, kmaps):
    db = os.path.join(db_dir, db_id, db_id + ".sqlite")
    schema = Schema(get_schema(db))
    g_sql = get_sql(schema, g_str)

    try:
        p_sql = get_sql(schema, p_str)
    except:
        # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
        p_sql = {
            "except": None,
            "from": {"conds": [], "table_units": []},
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [False, []],
            "union": None,
            "where": [],
        }

    # rebuild sql for value evaluation
    kmap = kmaps[db_id]
    g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
    exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql)
    return exec_score

def get_top_k_accuracies(top_k_sqls, gold_sqls, db_ids, db_dir, table_path, kmaps):
    print('\n Getting Top-k accuracies \n')
    best_em_results = []
    best_ex_results = []
    for top_sqls, gold_sql, db_id in zip(top_k_sqls, gold_sqls, db_ids):
        em_results = [evaluate_single(
                                    gold_sql,
                                    q,
                                    db_id,
                                    db_dir=db_dir,
                                    table_file=table_path)
                        for q in top_sqls]
        ex_results = [int(eval_exec_single(gold_sql, q, db_id, db_dir, kmaps)) for q in top_sqls]
        best_em_results.append(max(em_results))
        best_ex_results.append(max(ex_results))

    return best_em_results, best_ex_results

def get_reranker_accuracies(pred_sqls, gold_sqls, db_ids, db_dir, table_path, kmaps):
    #print('\nObtaining reranker performance\n')
    em_results = [evaluate_single(
                                gold_sql,
                                pred_sql,
                                db_id,
                                db_dir=db_dir,
                                table_file=table_path)
                    for gold_sql, pred_sql, db_id in zip(gold_sqls,pred_sqls,db_ids)]
    ex_results = [int(eval_exec_single(gold_sql, pred_sql, db_id, db_dir, kmaps)) 
                    for gold_sql, pred_sql, db_id in zip(gold_sqls,pred_sqls,db_ids)]
    return em_results, ex_results
        


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir",type=str)
    parser.add_argument("--dev_path", type=str, default="dataset/dev.json")
    parser.add_argument("--table_path", type=str, default="dataset/tables.json")
    parser.add_argument("--dataset_path", type=str, default="dataset/database")
    parser.add_argument("--output", type=str, default="predictions_with_vals_fixed4.txt")
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_gold", type=str, default=None)
    args = parser.parse_args()

    overrides = {
        "dataset_reader": {
            "tables_file": args.table_path,
            "dataset_path": args.dataset_path,
        }
    }
    overrides["validation_dataset_reader"] = {
        "tables_file": args.table_path,
        "dataset_path": args.dataset_path,
    }

    if args.cases is not None:
        cases = pickle.load(open(args.cases,"rb"))
        # cases = [cases[10], cases[19]]
        # cases = cases[9:10]
    else:
        cases = None

    archive_dir = args.archive_dir
    predictor = Predictor.from_path(
        archive_dir, cuda_device=args.gpu, overrides=overrides
    )
    print("after pred")

    final_beam_acc_stats = []
    leaf_acc_stats = []
    reranker_acc_stats = []
    spider_acc_stats = []
    ll_leafs = []
    mrr = []
    ma_precision = []
    gold_thresholds = []
    gold_thresholds_mask = []
    level_wise_recall_stats = defaultdict(list)
    top_k_sqls = []
    gold_sqls = []
    pred_sqls = []
    db_ids = []
    questions = []
    decoder_timesteps = 9

    questions_path = args.dev_path.replace('.pkl', '.ques')
    questions = [line.strip() for line in open(questions_path)]
    kmaps = build_foreign_key_map_from_json(args.table_path)
    

    if args.output_gold is not None:
        f_output_gold = open(args.output_gold,'w')
    else:
        f_output_gold = None

    with open(args.output, "w") as g:    
        with open(args.dev_path,'rb') as f:
            dev_pkl = pickle.load(f)
            for i, el in enumerate(tqdm.tqdm(dev_pkl)):
                if i == 0:
                    el_0 = el
                instance = el
                if i == 0:
                    instance_0 = instance
                
                if instance is not None and cases is not None:
                    instance = update_instance_with_cases(instance, cases, i)

                # used for gold retrieved pickles
                if "cases" in instance.fields:
                    # used for gold retrieved pickles
                    instance = add_cases_to_instance(instance)
                    el = instance

                if instance is not None:
                    predictor._dataset_reader.apply_token_indexers(instance)
                    if cases is not None:
                        #print('cases is not None')
                        with torch.cuda.amp.autocast(enabled=True):
                            out = predictor._model.forward_on_instances(
                                [instance]
                            )
                            pred = out[0]["sql_list"]
                            db_id = el['db_id'].metadata
                            gold_sql = el['gold_sql'].metadata
                    elif isinstance(instance['db_id'],ListField):
                        with torch.cuda.amp.autocast(enabled=True):
                            out = predictor._model.forward_on_instances(
                                [instance]
                            )
                            pred = out[0]["sql_list"]
                            db_id = list(el['db_id'])[0].metadata
                            gold_sql = list(el['gold_sql'])[0].metadata
                    else:
                        with torch.cuda.amp.autocast(enabled=True):
                            out = predictor._model.forward_on_instances(
                                [instance, instance_0]
                            )
                            pred = out[0]["sql_list"]
                            db_id = el['db_id'].metadata
                            gold_sql = el['gold_sql'].metadata
                else:
                    pred = "NO PREDICTION"
                if out[0]['final_beam_acc'] == True:
                    final_beam_acc_stats.append(1)
                else:
                    final_beam_acc_stats.append(0)
                if out[0]['reranker_acc'] == True:
                    reranker_acc_stats.append(1)
                else:
                    reranker_acc_stats.append(0)
                if out[0]['spider_acc'] == True:
                    spider_acc_stats.append(1)
                else:
                    spider_acc_stats.append(0)
                #print(f'final_beam_acc: {out[0]["final_beam_acc"]}, leaf_acc: {out[0]["leaf_acc"]}')
                if out[0]['leaf_acc'] == True:
                    leaf_acc_stats.append(1)
                else:
                    leaf_acc_stats.append(0)

                if "leaf_log" in out[0]:
                    ll_leafs.append(out[0]['leaf_log'].cpu().item())
                if "inv_rank" in out[0]:
                    mrr.append(out[0]['inv_rank'].cpu().item())
                if "avg_prec" in out[0]:
                    ma_precision.append(out[0]["avg_prec"].cpu().item())

                for decoding_step in range(decoder_timesteps):
                    step_key = f'level_wise_recall_{decoding_step+1}'
                    if step_key in out[0]:
                        step_value = out[0][step_key]
                        level_wise_recall_stats[step_key].append(step_value)

                if "gold_threshold_list" in out[0]:
                    g_thresholds = out[0]["gold_threshold_list"]
                    g_th_mask = (g_thresholds>=-1000) * (g_thresholds<=1000)
                    g_thresholds[~g_th_mask] = 0.0
                    gold_thresholds.append(g_thresholds)
                    gold_thresholds_mask.append((g_thresholds>=-1000) * (g_thresholds<=1000))

                if 'top_k_sql_list' in out[0]:
                    top_k_sqls.append(out[0]['top_k_sql_list'])

                gold_sqls.append(gold_sql)
                db_ids.append(db_id)

                g.write(f"{pred}\t{db_id}\n")
                if f_output_gold is not None:
                    f_output_gold.write(f"{gold_sql}\t{db_id}\n")

                pred_sqls.append(pred)

    if f_output_gold is not None:
        f_output_gold.close()
    
    print()
    print()
    print(f'Mean final_beam_acc: {np.mean(final_beam_acc_stats)}')
    print(f'Mean reranker_acc: {np.mean(reranker_acc_stats)}')
    print(f'Mean spider EM: {np.mean(spider_acc_stats)}')
    #print(f'final_beam_acc_stats: {final_beam_acc_stats}')
    print(f'Mean leaf_acc: {np.mean(leaf_acc_stats)}')

    em_results, ex_results = get_top_k_accuracies(top_k_sqls, gold_sqls, db_ids, 
                                                args.dataset_path, args.table_path, kmaps)
    
    print(f'Mean top-k beam EM: {np.mean(em_results)}')
    print(f'Mean top-k beam EX: {np.mean(ex_results)}')

    original_em, original_ex = get_reranker_accuracies(pred_sqls, gold_sqls, db_ids,
                                args.dataset_path, args.table_path, kmaps)

    print(f'Mean Original EM: {np.mean(original_em)}')
    print(f'Mean Original EX: {np.mean(original_ex)}')
    print()
    print()


if __name__ == "__main__":
    main()
