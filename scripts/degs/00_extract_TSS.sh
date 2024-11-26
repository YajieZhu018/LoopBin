#!/bin/bash
# extract TSS from gtf file
ol=/usr/users/yzhu1/Genome/hg38/
gtf=/usr/users/yzhu1/Genome/hg38/hg38.ncbiRefSeq.filtered.gtf
# extend 5kb from the start of transcript
awk 'BEGIN {OFS=FS="\t"}
    $3=="transcript" {
        # get gene name
        split($9,a,"\""); gene=a[2]
        if ($7=="+" && $4>5000) print $1,$4-5000,$4,$7,gene
        if ($7=="-") print $1,$5,$5+5000,$7,gene
    }' $gtf > temp.hg38.ncbiRefSeq.5kb.upstreamTSS.bed
sort -u temp.hg38.ncbiRefSeq.5kb.upstreamTSS.bed > "$ol"hg38.ncbiRefSeq.5kb.upstreamTSS.bed
rm temp.hg38.ncbiRefSeq.5kb.upstreamTSS.bed