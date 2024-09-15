#!/usr/bin/env bash

update_species_count () {
  gsed "/species_count/c species_count = $2" $1 \
    | gsed "/min_dist = /c\	min_dist = 0.5" \
    | gsed "s/\r//g" \
    > initial.mtp
}

mlip=~/codes/mlip-2
mlp=$mlip/bin/mlp
root=$PWD
original=$(realpath $(dirname $0))/../../original/molecules
# species_counts=([762]=1 [291]=2 [14214]=2 [23208]=1)
species_counts=([23208]=1)

for molecule in ${!species_counts[@]}; do
  # for level in 02 04 06 08 10; do
  for level in 10; do
    echo $molecule
    echo $level
    d=$molecule/$level
    mkdir -p $d
    cd $d
    update_species_count $mlip/untrained_mtps/$level.mtp ${species_counts[$molecule]}
    $mlp train \
      --init-params=same \
      initial.mtp \
      $original/$molecule/training.cfg \
      --trained-pot-name=pot.mtp
    $mlp calc-efs \
      pot.mtp \
      $original/$molecule/training.cfg \
      out.cfg
    cd $root
  done
done
