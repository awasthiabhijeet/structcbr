import pickle
import random
import os
import json
import allennlp
from tqdm import tqdm
import sys

SEED = int(sys.argv[1])
random.seed(SEED)

NUM_FEWSHOT = 30
NUM_VAL = 1 

def augment_question_with_schema(questions, instances):
    is_gold_leafs = [item['is_gold_leaf'].tensor.numpy()[14:] for item in instances]
    entities = [item['entities'].metadata[14:] for item in instances]
    assert len(is_gold_leafs) == len(entities)
    selected_elements = [[entities[i][j] for j in range(len(entities[i])) if is_gold_leafs[i][j]==1]  
                        for i in range(len(is_gold_leafs))]
    questions = [q+'\n'+' '.join(s)+'\n\n' for q,s in zip(questions, selected_elements)]
    return questions


PICKLE_DIR="processed_data"
SPLIT_DIR = os.path.join(PICKLE_DIR,'spider_val_cbr_splits_30',f'split_{SEED}')
VAL_PICKLE = os.path.join(PICKLE_DIR,'val_original.pkl')
VAL_SQLS = "dataset/dev_gold_filtered.sql"
VAL_DATA = "dataset/dev.json"
INVALID_INDICES = [175, 926, 927]
val_data = json.load(open(VAL_DATA))
val_data = [item for i,item in enumerate(val_data) if i not in INVALID_INDICES]
val_questions = [item["question"] for item in val_data]


DB_IDS=['world_1', 'car_1', 'cre_Doc_Template_Mgt', 'dog_kennels', 'flight_2']
db_instances = dict((key,[]) for key in DB_IDS)
db_sqls = dict((key,[]) for key in DB_IDS)
db_questions = dict((key,[]) for key in DB_IDS)
db_data = dict((key,[]) for key in DB_IDS)
val_instances = pickle.load(open(VAL_PICKLE,"rb"))
val_sqls = [line for line in open(VAL_SQLS)]
assert len(val_instances) == len(val_sqls) == len(val_questions)

for item,sql,question,data in zip(val_instances, val_sqls, val_questions, val_data):
    db_id = item['db_id'].metadata
    if db_id in DB_IDS:
        db_instances[db_id].append(item)
        db_sqls[db_id].append(sql)
        db_questions[db_id].append(question)
        db_data[db_id].append(data)

for db_id in DB_IDS:
    print(db_id)
    out_dir = os.path.join(SPLIT_DIR,db_id)
    os.makedirs(out_dir, exist_ok=True)
    instances = db_instances[db_id]
    sqls = db_sqls[db_id]
    questions = db_questions[db_id]
    data = db_data[db_id]
    zip_instances_sqls = list(zip(instances,sqls,questions, data))
    random.shuffle(zip_instances_sqls)
    instances, sqls, questions, data = zip(*zip_instances_sqls)
    instances, sqls, questions, data = list(instances), list(sqls), list(questions), list(data)
    train_instances = instances[0:NUM_FEWSHOT]
    val_instances = instances[-NUM_VAL:]
    test_instances = instances[NUM_FEWSHOT:-NUM_VAL]
    train_sqls = sqls[0:NUM_FEWSHOT]
    val_sqls = sqls[-NUM_VAL:]
    test_sqls = sqls[NUM_FEWSHOT:-NUM_VAL]
    train_questions = questions[0:NUM_FEWSHOT]
    val_questions = questions[-NUM_VAL:]
    test_questions = questions[NUM_FEWSHOT:-NUM_VAL]
    train_data = data[0:NUM_FEWSHOT]
    val_data = data[-NUM_VAL:]
    test_data = data[NUM_FEWSHOT:-NUM_VAL]
    assert len(train_sqls) == len(train_instances) == len(train_questions)
    assert len(val_sqls) == len(val_instances) == len(val_questions)
    assert len(test_sqls) == len(test_instances) == len(test_questions)
    assert len(test_data) == len(test_data) == len(test_data)
    assert len(train_instances) + len(val_instances) + len(test_instances) == len(instances)
    train_pkl = os.path.join(out_dir,'train.pkl')
    val_pkl = os.path.join(out_dir, 'val.pkl')
    test_pkl = os.path.join(out_dir, 'test.pkl')
    pickle.dump(train_instances, open(train_pkl,"wb"))
    pickle.dump(val_instances, open(val_pkl,"wb"))
    pickle.dump(test_instances, open(test_pkl,"wb"))
    sql_train = os.path.join(out_dir, 'train.sql')
    sql_val = os.path.join(out_dir, 'val.sql')
    sql_test = os.path.join(out_dir, 'test.sql')
    open(sql_train,'w').writelines(train_sqls)
    open(sql_val,'w').writelines(val_sqls)
    open(sql_test,'w').writelines(test_sqls)
    train_ques = os.path.join(out_dir,'train.ques')
    val_ques = os.path.join(out_dir, 'val.ques')
    test_ques = os.path.join(out_dir, 'test.ques')
    open(train_ques,'w').writelines([item+'\n' for item in train_questions])
    open(val_ques,'w').writelines([item+'\n' for item in val_questions])
    open(test_ques,'w').writelines([item+'\n' for item in test_questions])
    train_json = os.path.join(out_dir,'train.json')
    val_json = os.path.join(out_dir, 'val.json')
    test_json = os.path.join(out_dir, 'test.json')
    json.dump(train_data, open(train_json,"w"), indent=4)
    json.dump(val_data, open(val_json,"w"), indent=4)
    json.dump(test_data, open(test_json,"w"), indent=4)