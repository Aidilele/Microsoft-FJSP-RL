# Simultaneous Generation and Improvement: A Unified RL Paradigm for FJSP Optimization

---

## overview

---

<img src="assets/overview.svg" alt="Alt text" title="Optional title" style="">


## Running

---
Creating the scripts FSJP instance data by running `./dataprocess/CreatInstance`.

    def main():
        '''
        Generate Random instance by parameters
        '''
        batch_size = 50
        num_jobs = 20
        num_mas = 5
        opes_per_job_min = int(num_mas * 0.8)
        opes_per_job_max = int(num_mas * 1.2)
        case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max,
                             flag_same_opes=False, flag_doc=True,
                             path='../dataset/2005test/')
        for each_case_index in range(batch_size):
            case.get_case(each_case_index)

Training model by running `./DuelingDQN_Train_main.py`. And all of the training parameters can be modified in file `./config.json`.
All of the training data will be saved in `./runs`.

Evaluating the model by running `./evaluate.py`. Experiments Ganttchart visualization will be saved in `./render_result`.

## FJSP simulator

Users can  implemente their customize algorithm by using our the FJSP environment.All of the dependent files are included in directory `./env` and `./utils`.
You can create a FSJP environment by running:

    dataset = '1005'    #the dataset must be located in ./dataset
    env = FJSPEnviroment(dir_path='./dataset/' + dataset)
    env.reset()

more functions about simulator can find in `./env/Environment.py`
