#!/bin/bash

config=$1
id=$2
test=$3
params=${*:4}

DEBUG=true python -m src.main +experiment=${config} +test=${test} hydra.run.dir=./outputs/${config}/${id} mode=test hydra.job.name=${mode} wandb.activated=false ${params}
