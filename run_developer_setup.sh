#!/bin/bash

./unix_build.sh

./clean.sh
python setup.py clean

pip install -r requirements.txt

pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip

python setup.py develop

pip install -e .
