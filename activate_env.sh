# CURRENT_ENV="narcolepsy-detector"
# CUDNN_VER="7.0.4"

# if [ $CONDA_DEFAULT_ENV != $CURRENT_ENV ]; then
#     echo "Unloading current environment"
#     conda deactivate
#     echo "Activating environment: $CURRENT_ENV"
#     conda activate $CURRENT_ENV
# else
#     echo "Environment already active"
# fi

# echo "Loading cuDNN version $CUDNN_VER"
# ml load cudnn/7.0.4

# echo "Environment ready!"

CURRENT_ENV="nml"
# TF_VER="2.4.1_py36"
CUDA_VER="11.2.0"
CUDNN_VER="8.1.1.33"

# echo "Loading TF version $TF_VER"
# ml load py-tensorflow/$TF_VER

source $GROUP_HOME/miniconda3/bin/activate
if [ $CONDA_DEFAULT_ENV != "base" ]; then
    echo "Unloading current environment"
    conda deactivate
    echo "Activating environment: $CURRENT_ENV"
    conda activate $CURRENT_ENV
else
    echo "Activating environment: $CURRENT_ENV"
    conda activate $CURRENT_ENV
fi

echo "Loading CUDA version $CUDA_VER"
ml load cuda/$CUDA_VER

echo "Loading CUDNN version $CUDNN_VER"
ml load cudnn/$CUDNN_VER

echo "Environment ready!"
