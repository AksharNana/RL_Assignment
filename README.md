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