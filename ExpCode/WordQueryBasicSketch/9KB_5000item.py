import os
import sys
import argparse
import multiprocessing
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

from SourceCode.Factory import Factory

# CUDA_LAUNCH_BLOCKING = 1

def parse_command_line(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true', default=False)
    parser.add_argument('--pass_cuda', action='store_true', default=False)
    parser.add_argument('--pass_cpu', action='store_true', default=False)
    args = parser.parse_args()
    prod = args.prod
    pass_cuda = args.pass_cuda
    pass_cpu = args.pass_cpu
    if prod:
        print('working in prod, set big batch size and eval gap:')
        config['train_config']['queue_size'] = 20
        config['logger_config']['flush_gap'] = 10

        config['train_config']['lr'] = 0.0001
        config['logger_config']['test_task_group_size'] = 10
        config['logger_config']['eval_gap'] = 10000
    else:
        print('working in dev, set small hyperparameter for less memory consume')
    print('cuda num',config['train_config']['cuda_num'])
    print('queue_size', config['train_config']['queue_size'])
    print('eval_gap', config['logger_config']['eval_gap'])
    print('lr ',config['train_config']['lr'])
    print('test_task_item_size_list', config['logger_config']['test_task_item_size_list'])
    print('memory size:', config['dim_config']['depth_dim'] * config['dim_config']['embedding_dim'] * config['dim_config'][
              'slot_dim'] * 4 / 1024, 'KB')
    print('support_set_item_upper:', config['data_config']['item_upper'])
    print('support_set_item_lower:', config['data_config']['item_lower'])
    if pass_cuda:
        pass_cuda_tensor = True
        print('passing gpu tensor,attention: must in linux!!')

    elif pass_cpu:
        pass_cuda_tensor = False
        print('passing cpu tensor')

    else:
        if not prod:
            pass_cuda_tensor = False
            print('working in pc environment, passing cpu tensor')
        else:
            pass_cuda_tensor = True
            print('working in prod environment ,passing gpu tensor,attention: must in linux!!')
    return prod, pass_cuda_tensor

def init_config():
    data_config = {
        "train_comment": "9KB_5000item",
        # Optional: WordQueryBasicSketch
        "dataset_name": 'WordQueryBasicSketch',
        'dataset_path': '../../Dataset/WordQueryTaskFile/',
        'item_upper': 5000,
        'item_lower': 2,
        'skew_lower': 1,
        'skew_upper': 10,
        "zipf_param_upper":1.3,
        "zipf_param_lower":0.8,
    }
    dim_config = {
        "input_dim": 60,
        "embedding_dim": 23,
        "refined_dim": 5,
        "slot_dim": 50,
        "depth_dim": 2,
    }
    train_config = {
        "train_step": 5000000,
        "lr": 0.0001,
        "cuda_num": 0,
        'queue_size': 20,
    }
    logger_config = {
        "flush_gap": 1,
        "save_gap": 10000,
        "eval_gap": 500,
        "test_task_item_size_list": [1000, 2000, 3000, 4000, 5000, 6000, 7000,8000,9000,10000],
        "test_task_group_size": 10,
        "test_zipf_param_list":[0.5,1.1,1.0,1.5],
    }
    factory_config = {
        # Optional: LossFunc_for_ARE_AAE,LossFunc_for_MSE_ARE
        "loss_class": "LossFunc_for_MSE_ARE",
        # Optional: BasicMemoryMatrix
        "memory_calss": "BasicMemoryMatrix",
        "attention_class": "AttentionMatrix",
        # "sparse_degree": 2,
        "decode_weight_class": "WeightDecodeNetResidual",
        # optional: Model
        "model_class": "Model",
    }
    hidden_layer_config = {
        "embedding_hidden_layer_size": 64,
        "refined_hidden_layer_size": 32,
        "decode_hidden_layer_size": 256,
    }
    config = {
        "train_config": train_config,
        "factory_config": factory_config,
        "dim_config": dim_config,
        "hidden_layer_config": hidden_layer_config,
        "data_config": data_config,
        "logger_config": logger_config
    }
    return config

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    config = init_config()
    train_config = config['train_config']
    prod, pass_cuda_tensor = parse_command_line(config=config)
    factory = Factory(config)
    MGS = factory.init_MGS(prod)
    MGS.train(train_config["train_step"], pass_cuda_tensor, train_config['queue_size'])
