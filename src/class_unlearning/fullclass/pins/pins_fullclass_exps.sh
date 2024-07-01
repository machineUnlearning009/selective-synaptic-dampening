# Task bash files to be called by the overall bash file for all experiments of the specified model type.
# We provide all of the individual bash files per task and model to make individual experiments easy to call.

# Get the directory of the current script
script_dir=$(dirname $(realpath $0))

# Determine the root project directory (adjust the relative path as needed)
root_dir=$(realpath "${script_dir}/../../..")

# Set the PYTHONPATH to include the root project directory
export PYTHONPATH=${root_dir}:${PYTHONPATH}

reset_cuda(){
    sleep 10
}

DEVICE=$1
seed=$2
#############################################################
################ CIFAR100 ROCKET FORGETTING #################
#############################################################
declare -a StringArray=("1" "10" "20" "30" "40") # IDs of faces to forget


dataset=PinsFaceRecognition
n_classes=105
weight_path=checkpoint/ResNet18/Saturday_12_August_2023_10h_50m_04s/ResNet18-PinsFaceRecognition-194-best.pth
script_path="${script_dir}/../forget_full_class_main.py"

for val in "${StringArray[@]}"; do
    forget_class=$val
    # Run the Python script
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method baseline -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method ssd_tuning -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method finetune -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method amnesiac -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method blindspot -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method UNSIR -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python $script_path -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method retrain -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda
done
