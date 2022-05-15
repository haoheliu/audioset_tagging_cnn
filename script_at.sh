######################## ENVIRONMENT ########################
eval "$('/vol/research/dcase2022/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate panns
which python
cd /vol/research/dcase2022/project/audioset_tagging_cnn_for_bbc

######################## SETUP ########################
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
# wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
MODEL_TYPE="Cnn14"
echo $(pwd)

######################## RUNNING ENTRY ########################
python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --path=$1\
    --cuda

# CHECKPOINT_PATH="Cnn14_DecisionLevelMax_mAP=0.385.pth"
# wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1
# MODEL_TYPE="Cnn14_DecisionLevelMax"

# python3 pytorch/inference.py sound_event_detection \
#     --model_type=$MODEL_TYPE \
#     --checkpoint_path=$CHECKPOINT_PATH \
#     --cuda