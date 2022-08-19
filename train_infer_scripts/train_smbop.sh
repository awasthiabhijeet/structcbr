python3 -u exec.py \
  --config_path "configs/roberta-pkl-defaults.jsonnet" \
  --rat_layers 4 \
  --grad_acum 2 \
  --batch_size 40 \
  --name "smbop_base"