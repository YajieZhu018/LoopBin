#!/bin/bash
# Verification of the three argument
target="$1"
destination="$2"
resolution="$3"
chromosome="$4"

if [ $# -gt 5 ] || [ $# -lt 5 ];then
    echo "Missing argument. Put filetarget, folder destination, resolution, chromosome, name of the protein"
    exit 1
fi
# To keep the size of each chromosome for the genome mm10 (mouse: chr1-19 + X)
declare -A chrom
chrom[chr1]=195471971;chrom[chr2]=182113224;chrom[chr3]=160039680;chrom[chr4]=156508116;chrom[chr5]=151834684;chrom[chr6]=149736546
chrom[chr7]=145441459;chrom[chr8]=129401213;chrom[chr9]=124595110;chrom[chr10]=130694993;chrom[chr11]=122082543;chrom[chr12]=120129022
chrom[chr13]=120421639;chrom[chr14]=124902244;chrom[chr15]=104043685;chrom[chr16]=98207768;chrom[chr17]=94987271;chrom[chr18]=90702639;chrom[chr19]=61431566
chrom[chrX]=171031299

# Argument input
target="$1"
destination="$2"
resolution="$3"
chromosome="$4"
# for the path of each files
name_len=$(( $resolution  / 1000 ))"K"
name_bed=$destination"/"$chromosome"_"$5$name_len".bed"
name_bedgraph=$destination"/"$chromosome"_"$5$name_len".bedgraph"
name_tmp=$destination"/"$chromosome$5"temp.txt"

# Create bed file
n=$(( ${chrom[$4]} / $resolution ))
start=0
end=$((n * resolution))
step=$resolution
> "$name_bed"
for (( c=1; start<end; c++, start+=step ))
do
    echo -e "$4\t$start\t$((start + step))\t$4_$c"
done > "$name_bed"
# get bigwig signal over the bed region
bigWigAverageOverBed $1 $name_bed $name_tmp

# make bedgraph
awk 'BEGIN {OFS=FS="\t"}
    NR==FNR {dic[$1]=$5;next}
    $4 in dic {print $1,$2,$3,dic[$4]}' $name_tmp $name_bed > $name_bedgraph
# delete temp file
rm -fr $name_tmp
rm -fr $name_bed