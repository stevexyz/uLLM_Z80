#!/bin/bash
cat examples/tinychat/training-data.txt | ./feedme.py --epochs 300 --chat
./buildz80com.py -m command_model_autoreg.pt -o CHAT.COM
