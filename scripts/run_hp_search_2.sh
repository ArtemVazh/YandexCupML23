cd ..;
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=./config.yaml python main.py model.nhead=12 model.num_layers=6 data.aug=0.6 data.max_length=80 data.k_folds=5 data.batch_size=128 training.epochs=100 training.lr=3e-05 training.weight_decay=1e-05 training.T_0=10 training.eta_min=1e-07
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=./config.yaml python main.py model.nhead=12 model.num_layers=6 data.aug=0.6 data.max_length=80 data.k_folds=5 data.batch_size=128 training.epochs=100 training.lr=5e-05 training.weight_decay=1e-05 training.T_0=10 training.eta_min=1e-07
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=./config.yaml python main.py model.nhead=12 model.num_layers=9 data.aug=0.6 data.max_length=80 data.k_folds=5 data.batch_size=128 training.epochs=100 training.lr=3e-05 training.weight_decay=1e-05 training.T_0=10 training.eta_min=1e-07
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=./config.yaml python main.py model.nhead=12 model.num_layers=9 data.aug=0.6 data.max_length=80 data.k_folds=5 data.batch_size=128 training.epochs=100 training.lr=5e-05 training.weight_decay=1e-05 training.T_0=10 training.eta_min=1e-07