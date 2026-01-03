#!/bin/bash
if [ "$1" != "--justrun" ]
  then
    cat examples/tinychat/training-data.txt | ./feedme.py --epochs 1 --chat
fi
./buildz80com.py -m command_model_autoreg.pt -o CHAT.COM
