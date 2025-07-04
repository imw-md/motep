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
original=$(realpath $(dirname $0))/../../original/crystals
# crystals=(cubic noncubic)
crystals=(size)

for crystal in $crystals; do
  species_count=1
  if [ $crystal = 'multi' ]; then
    species_count=3
  fi
  for level in 02 04 06 08 10; do
    echo $crystal
    echo $level
    d=$crystal/$level
    mkdir -p $d
    cd $d
    update_species_count $mlip/untrained_mtps/$level.mtp ${species_count}
    $mlp train \
      --init-params=same \
      initial.mtp \
      $original/$crystal/training.cfg \
      --trained-pot-name=pot.mtp
    $mlp calc-efs \
      pot.mtp \
      $original/$crystal/training.cfg \
      out.cfg
    cd $root
  done
done
