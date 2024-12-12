#/bin/bash
# find unique loops in each group
# input folder
folder=$1
il="$folder"intersect/01_sorted/
# output folder
ol="$folder"unique/01_unique/
mkdir -p $ol
# find unique loops in each group
# get degron unique
file1="$il"degron_labels_loops.bedpe
file2="$il"control_labels_loops.bedpe
pgltools intersect -v -u -d 8000 -a $file1 -b $file2 | awk 'BEGIN {OFS=FS="\t"} {print $1,$2,$3,$4,$5,$6,$9}' > "$ol"degron_unique_labels_loops.bedpe
# get control unique
pgltools intersect -v -u -d 8000 -a $file2 -b $file1 | awk 'BEGIN {OFS=FS="\t"} {print $1,$2,$3,$4,$5,$6,$11}' > "$ol"control_unique_labels_loops.bedpe
