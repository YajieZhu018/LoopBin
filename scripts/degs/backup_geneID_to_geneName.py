from Bio import Entrez
import pandas as pd
# set email
Entrez.email = "yajie.zhu@icloud.com"
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/degs/'
# loop files
loop_file = f'{ol}'
# FPKM with geneID
fpkm_file = '/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_norm_counts_FPKM_GRCh38.p13_NCBI.tsv'
# read the file into dataframe
fpkm_df = pd.read_csv(fpkm_file, sep='\t')
geneID = fpkm_df['GeneID'].tolist()

import concurrent.futures
import urllib.error

def fetch_gene_name(gene_id):
    try:
        handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
        records = Entrez.read(handle)
        return records[0]["Entrezgene_gene"]["Gene-ref"]["Gene-ref_locus"]
    except (urllib.error.HTTPError, urllib.error.URLError, Exception):
        return "NA"

# Fetch gene names in parallel using multiple threads
with concurrent.futures.ThreadPoolExecutor() as executor:
    gene_names = executor.map(fetch_gene_name, geneID)

# Convert the generator to a list
geneName = list(gene_names)

# add geneName to the dataframe
fpkm_df['geneName'] = geneName
# save the dataframe
out_file = '/usr/users/yzhu1/LoopBin/data/RNA-seq/GSE176285_norm_counts_FPKM_GRCh38.p13_NCBI_geneName.tsv'
fpkm_df.to_csv(out_file, sep='\t', index=False)