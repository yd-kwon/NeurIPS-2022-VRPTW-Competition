#!/bin/bash

#pip install -r requirements.txt

cd hgs 
make clean
make all
cd ..

cd hgs_dynamic
make clean
make all
cd ..

