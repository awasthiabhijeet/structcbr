ARCHIVE_DIR='path/to/pre-trained/smbop/ckpt'
for split_size in 30 20
do
  for split_id in 2 1 0
  do
    for DB_NAME in world_1 car_1 cre_Doc_Template_Mgt dog_kennels flight_2
    do
      SPIDER_PICKLE_DIR=processed_data/spider_val_cbr_splits_"$split_size"/split_"$split_id"

      output_preds=$ARCHIVE_DIR/beam_spider_"$DB_NAME"_SPLIT_"$split_id"_CS_"$split_size"_test_pred.sql
      output_eval=$ARCHIVE_DIR/beam_spider_"$DB_NAME"_SPLIT_"$split_id"_CS_"$split_size"_test_pred.eval
      output_infer=$ARCHIVE_DIR/beam_spider_"$DB_NAME"_SPLIT_"$split_id"_CS_"$split_size"_test_pred.infer
      output_gold=$ARCHIVE_DIR/beam_spider_"$DB_NAME"_SPLIT_"$split_id"_CS_"$split_size"_test_pred.gold
      
      echo 
      echo $DB_NAME $split_id $split_size
      echo
      # python3 -u large_knn_eval_kaggledb_beam.py \
      python3 -u eval_beam.py \
        --archive_dir=$ARCHIVE_DIR \
        --dev_path=$SPIDER_PICKLE_DIR/$DB_NAME/test.pkl \
        --table_path=dataset/tables.json \
        --dataset_path=dataset/database \
        --output=$output_preds \
        --output_gold=$output_gold > $output_eval

      echo >> $output_eval 
      echo >> $output_eval 

      python3 -u smbop/eval_final/evaluation.py \
        --gold $SPIDER_PICKLE_DIR/$DB_NAME/test.sql  \
        --pred $output_preds \
        --etype all \
        --db  dataset/database  \
        --table dataset/tables.json >> $output_eval
    done
  done
done