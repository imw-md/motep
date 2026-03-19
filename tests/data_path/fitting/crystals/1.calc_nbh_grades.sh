#!/usr/bin/env bash

mlip=~/codes/mlip-3
mlp=$mlip/bin/mlp
root=$PWD
original=$(realpath $(dirname $0))/../../original/crystals
# crystals=(cubic noncubic)
crystals=(multi)

for crystal in $crystals; do
  # for level in 02 04 06 08 10; do
  for level in 10; do
    echo $crystal
    echo $level
    d=$crystal/$level
    mkdir -p $d
    pushd $d
    $mlp train \
      --save_to=nbh.almtp \
      --iteration_limit=0 \
      --no_mindist_update=true \
      --al_mode=nbh \
      pot.mtp \
      $original/$crystal/training.cfg
    $mlp calculate_grade \
      nbh.almtp \
      out.cfg \
      tmp.cfg
    mv tmp.cfg.0 out.cfg
    rm nbh.almtp
    popd
  done
done
