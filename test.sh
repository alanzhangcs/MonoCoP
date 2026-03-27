GPUS=${@:1}
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=$GPUS  python train_val.py ${PY_ARGS} -e