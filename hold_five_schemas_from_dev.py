import json

EVAL_SCHEMAS = ['world_1', 'car_1', 'cre_Doc_Template_Mgt', 'dog_kennels', 'flight_2']

dev_data = json.load(open('dataset/dev.json'))
remaining_dev_path = 'dataset/remaining_dev.json'
eval_dev_path = 'dataset/eval_dev.json'

eval_dev = []
remaining_dev = []

for item in dev_data:
	if item['db_id'] in EVAL_SCHEMAS:
		eval_dev.append(item)
	else:
		remaining_dev.append(item)

json.dump(eval_dev, open(eval_dev_path, 'w'))
json.dump(remaining_dev, open(remaining_dev_path, 'w'))