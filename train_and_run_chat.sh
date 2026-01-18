#!/bin/bash

# to activate torch python environment ". torchrun/bin/activate"

if [ "$1" == "--newrun" ]
  then
    rm command_model_autoreg.pt
fi

if [ "$1" != "--justrun" ]
  then
    cat examples/smschat/training-data.txt | ./feedme.py --epochs 300 --chat
fi
./buildz80com.py -m command_model_autoreg.pt -o CHAT.COM
#./buildfastz80com.py -m command_model_autoreg.pt -o CHATF.COM
