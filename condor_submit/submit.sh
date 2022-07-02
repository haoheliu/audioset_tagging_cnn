######################## ENVIRONMENT ########################
eval "$('/mnt/fast/nobackup/users/hl01486/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate panns
which python
cd /mnt/fast/nobackup/scratch4weeks/hl01486/project/audioset_tagging_cnn

######################## SETUP ########################
EXP_NAME="panns_cnn14"

######################## RUNNING ENTRY ########################
source scripts/4_train.sh