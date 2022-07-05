#!/bin/bash
#
#SBATCH -t 24:00:00
#
# filenames stdout and stderr - customise, include %j
#SBATCH -o result_%j.out
#SBATCH -e error_%j.err
#SBATCH -N 1
#SBATCH --tasks-per-node=20


source $HOME/.load_modules.sh
source $HOME/dune/venv/bin/activate
mpirun -N 1 python -c "import sys; print(sys.version); import dune"

cd $SNIC_TMP
mpirun -N 1 cp -r $HOME/dune-course/ .
cd dune-course

echo "STARTING"
mpirun python $@
echo "FINISHED"

ls

destination=$HOME/vtu/$SLURM_JOB_ID
mkdir -p $destination && cp -pr *vtu $destination
destination=$HOME/info/$SLURM_JOB_ID
mkdir -p $destination && cp *.json $destination

