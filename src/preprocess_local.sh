#!/bin/bash
target="$1"
name="$2"
folder="$3"
preprocessing_command="$4"
reso=8000

if [ "$target" = "empty" ]; then
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X
    do
        $preprocessing_command/empty_preprocess.sh $target $folder $reso chr$i $name &
    done
    wait
    exit
fi

for i in 1 2 3 4 5 6 7 8; do
    $preprocessing_command/preprocessing.sh $target $folder $reso chr$i $name &
done

# Attente de la fin de tous les processus enfants
wait

for i in 9 10 11 12 13 14 15 16
do
    $preprocessing_command/preprocessing.sh $target $folder $reso chr$i $name &
done

# Attente de la fin de tous les processus enfants
wait

for i in 17 18 19 20 21 22 X
do
    $preprocessing_command/preprocessing.sh $target $folder $reso chr$i $name &
done

# Attente de la fin de tous les processus enfants
wait