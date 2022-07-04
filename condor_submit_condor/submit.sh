######################## ENVIRONMENT ########################
eval "$('/vol/research/dcase2022/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate panns
which python
cd /vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn

######################## SETUP ########################
EXP_NAME="panns_cnn14"

######################## RUNNING ENTRY ########################
echo $1
source scripts/$1.sh