$ python3 train.py --save              --num-adversaries=1 --scenario=spies-like-us-1.4 --{save,load,plots,benchmark}-dir=$(pwd)/spies-like-us/experiments/1.4/
$ python3 train.py --restore --display --num-adversaries=1 --scenario=spies-like-us-1.4 --{save,load,plots,benchmark}-dir=$(pwd)/spies-like-us/experiments/1.4/ --max-episode-len=200

