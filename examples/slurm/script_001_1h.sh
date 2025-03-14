#!/usr/bin/env bash
#SBATCH -J script
#SBATCH --time 1:00:00
#SBATCH --export=HOME

source $HOME/.bashrc

# https://www.gnu.org/software/bash/manual/bash.html#Special-Parameters
# ($@) Expands to the positional parameters, starting from one.
$@
