#!/bin/bash

# define o endereço do nó mestre (onde o rank 0 será executado)
# trocar o localhost para o ip do mestre
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# numero total de processos que participarao do treinamento.
export WORLD_SIZE=4

# ativa o ambiente virtual
source venv/bin/activate

echo "Iniciando treinamento distribuído com PyTorch DDP..."

# comando para iniciar o treinamento distribuído
python3 -m torch.distributed.launch \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py

echo "Treinamento distribuído concluído."

deactivate