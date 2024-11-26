## To match the clusters output from two models of the same loop into a file: 
# chrom1, start1, end1, chrom2, start2, end2
# inputfile
file1=$1  #/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/prediction/control_rep1/labels_loops.bedpe
file2=$2  #/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/prediction/control_rep1/labels_loops.bedpe
# outputfile
outfile=$3  #/usr/users/yzhu1/LoopBin/trials/saved_models/vade_8clusters_control_rep1_CTCF_H3K27ac_H3K27me3_SMC1A_H3K4me1/cluster_shift/clusters_addd_with_7clusters_merged_control_degron_rep1.bedpe
# match two clusters
awk 'BEGIN{OFS=FS="\t"}
    FNR==NR {dic[$1"\t"$2"\t"$5]=$11; next;}
    $1"\t"$2"\t"$5 in dic {print $1,$2,$3,$4,$5,$6,dic[$1"\t"$2"\t"$5],$11}' $file1 $file2 > $outfile
