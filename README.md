## RL Assignment
To create the virtual environment:
```
pip3 install -U virtualenv
python3 -m virtualenv venv_grid2op
source venv_grid2op/bin/activate
pip3 install -r requirements.txt
```

### Tensorboard
Run with 
```
tensorboard --logdir=./tb_logs
```

### How to run and evaluate
1) In either DQN Agent or PPO Agent, open the final improvement folder as this would contain the latest model
    - Run the agent file using python agent_file_name.py
2) Once trained, run the test.py file.

OR 

If you only want to evaluate, you can just run the test.py file