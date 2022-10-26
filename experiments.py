import os
from utils.data_utils import load_numpy_arr
from utils.utils import test_model, print_result

def compare_models(exp_num):
    direct = 'first_experiment'
    names = ['teacher', 'mlp', 'student_mlp']
    results = []
    for mode in names:
        for i in range(1, exp_num+1):
            os.system(f"python main.py --mode={mode} --seed={i} --directory={direct}")
            if mode == 'teacher':
                os.system(f"python save_teacher_soft_labels.py --seed={i} --directory={direct}")
        results.append(test_model(name=mode, path=f'./data/{direct}/'))

    for mode, res in zip(names, results):
        print_result(mode, res)  

def mlp_2_layers_experiment(exp_num):
    direct = '2_layers_experiment'
    mode = 'mlp'
    for i in range(1, exp_num+1):
        os.system(f"python main.py --mode={mode} --seed={i} --directory={direct} --num_layers=2")
    res = test_model(name=mode, path=f'./data/{direct}/')
    print_result(mode + ' ' + direct, res)
    

def mlp_dropout_experiment(exp_num):
    direct = 'dropout_experiment'
    mode = 'mlp'
    for i in range(1, exp_num+1):
        os.system(f"python main.py --mode={mode} --seed={i} --directory={direct} --dropout_p=0.5")
    res = test_model(name=mode, path=f'./data/{direct}/')
    print_result(mode + ' ' + direct, res)
    

if __name__ == "__main__":
    exp_num = 10
    compare_models(exp_num)
    mlp_2_layers_experiment(exp_num)
    mlp_dropout_experiment(exp_num)
    
    
    