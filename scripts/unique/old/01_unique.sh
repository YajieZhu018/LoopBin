#/bin/bash
# find unique loops in each group
# input folder
folder=$1
il="$folder"intersect/01_sorted/
# output folder
ol="$folder"unique/01_sorted/
mkdir -p $ol
## merged loops of each condition but keep the label of each loop and add the label to the end of each line; the loop label is 1-6 
## loop through conditions
#for cond in control degron
#do
#    outfile="$ol""$cond"_loops_label_merged.bed
#    # loop through 1-6
#    for i in {1..6}
#    do
#        file="$il""$cond"_loops_label_"$i".bed
#        # print and add the label to the outfile
#        awk -v i="$i" '{print $0,i}' $file >> "$outfile"temp
#    done
#    # sort the merged file
#    pgltools sort "$outfile"temp > "$outfile"
#    rm "$outfile"temp
#done

# find unique loops in each group
for i in {1..6}
do
    # get degron unique
    file="$il"degron_loops_label_"$i".bed
    merged="$ol"control_loops_label_merged.bed
    pgltools intersect -v -u -d 8000 -a $file -b $merged > "$ol"degron_loops_label_"$i".bed
    # get control unique
    file="$il"control_loops_label_"$i".bed
    merged="$ol"degron_loops_label_merged.bed
    pgltools intersect -v -u -d 8000 -a $file -b $merged > "$ol"control_loops_label_"$i".bed
done
