#!/bin/bash
# Verification of the three argument
if [ $# -gt 5 ] || [ $# -lt 5 ];then
    echo "Missing argument. Put filetarget, folder destination, resolution, chromosome, name of the protein"
    exit 1
fi
# To keep the size of each chomosome for the genome hg38
declare -A chrom
chrom[chr1]=248956422;chrom[chr2]=242193529;chrom[chr3]=198295559;chrom[chr4]=190214555;chrom[chr5]=181538259;chrom[chr6]=170805979
chrom[chr7]=159345973;chrom[chr8]=145138636;chrom[chr9]=138394717;chrom[chr10]=133797422;chrom[chr11]=135086622;chrom[chr12]=133275309
chrom[chr13]=114364328;chrom[chr14]=107043718;chrom[chr15]=101991189;chrom[chr16]=90338345;chrom[chr17]=83257441;chrom[chr18]=80373285;chrom[chr19]=58617616;
chrom[chr20]=64444167;chrom[chr21]=46709983;chrom[chr22]=50818468;chrom[chrX]=156040895

# Argument input
target="$1"
destination="$2"
resolution="$3"
chromosome="$4"
# for the path of each files
name_len=$(( $resolution  / 1000 ))"K"
name_bed=$destination"/"$chromosome"_"$5$name_len".bedgraph"
name_tmp=$destination"/"$chromosome$5"temp.txt"
n=$(( ${chrom[$4]} / $resolution ))
start=0
end=$((n * resolution))
step=$resolution
> "$name_bed"
for (( c=1; start<end; c++, start+=step ))
do
    echo -e "$4\t$start\t$((start + step))\t0"
done > "$name_bed"