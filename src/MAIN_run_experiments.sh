# This is the main file to run all experiments
# Bash file to run different seeds (corresponding to the value) across all tasks
# Pass the GPU ID with the first parameter (e.g., 0; check via nvidia-smi)

#!/bin/bash
#set -e # uncomment to make the script stop when an error occurs; otherwise will ignore
DEVICE=$1


# TODO: Set the range for the number of seeds you want to run. Value is used as seed
# TODO: Do not forget to set the paths to the model weights in the other bash files (e.g., cifar20_fullclass_exps.sh)
# You might encounter issues with executing this file due to different line endings with Windows and Unix. Use dos2unix "filename" to fix.
for value in {1..123}
do  
    # ResNet18
    ./class_unlearning/subclass/cifar20/cifar20_subclass_exps.sh $DEVICE $value
    ./class_unlearning/random/cifar10/cifar10_random_exps.sh $DEVICE $value
    ./class_unlearning/fullclass/cifar20/cifar20_fullclass_exps.sh $DEVICE $value
    ./class_unlearning/fullclass/cifar100/cifar100_fullclass_exps.sh $DEVICE $value

     # Celebrity faces: www.kaggle.com/datasets/hereisburak/pins-face-recognition
     ./class_unlearning/pins_fullclass_exps.sh $DEVICE $value

    # Vision Transformer
    ./class_unlearning/random/cifar10/cifar10_random_exps_vit.sh $DEVICE $value
    ./class_unlearning/fullclass/cifar100/cifar100_fullclass_exps_vit.sh $DEVICE $value
    ./class_unlearning/fullclass/cifar20/cifar20_fullclass_exps_vit.sh $DEVICE $value
    ./class_unlearning/subclass/cifar20/cifar20_subclass_exps_vit.sh $DEVICE $value
    
   
done
