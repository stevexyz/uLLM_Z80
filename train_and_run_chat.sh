#!/bin/bash
if [ "$1" != "--justrun" ]
  then
    cat examples/smschat/training-data.txt | ./feedme.py --epochs 300 --chat
fi
./buildz80com.py -m command_model_autoreg.pt -o CHAT.COM
