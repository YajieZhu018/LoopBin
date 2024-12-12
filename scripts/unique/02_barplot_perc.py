# plot the percentage of each cluster in a barplot
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
# plot all loops
# input folder
folder = sys.argv[1]
il = f'{folder}/intersect/01_sorted/'
# output folder 
ol = f'{folder}/unique/02_plot/'
# make output folder
os.system(f'mkdir -p {ol}')
# create a dataframe to store the percentage of each cluster
df_all = pd.DataFrame()
# get the line number of the input file
# loop through control degron
for cond in ['control','degron']:
    file=f'{il}{cond}_labels_loops_formatted.bedpe'
    # read into a dataframe
    df = pd.read_csv(file,sep='\t',header=None)
    # count the number of each cluster; cluster label is in the 7th column
    df_count = df[6].value_counts()
    # sorted by cluster label and save to df_all
    df_all[cond] = list(df_count.sort_index())
# calculate the percentage of each cluster
df_all['control'] = df_all['control']/df_all['control'].sum()*100
df_all['degron'] = df_all['degron']/df_all['degron'].sum()*100
# plot the barplot
fig, ax = plt.subplots()
df_all.plot(kind='bar',ax=ax)
ax.set_ylim(0, 34)
plt.xlabel('Cluster')
plt.ylabel('Percentage (%)')
# make x labels 1-6
#plt.xticks(range(6),range(1,7),rotation='vertical')
plt.title('Percentage of all loops')
plt.savefig(f'{ol}barplot_all_percentage.pdf')  

# plot unique loops
# input folder
il = f'{folder}/unique/01_unique/'
# create a dataframe to store the percentage of each cluster
df_unique = pd.DataFrame()
# get the line number of the input file
# loop through control degron
for cond in ['control','degron']:
    file=f'{il}{cond}_unique_labels_loops.bedpe'
    # read into a dataframe
    df = pd.read_csv(file,sep='\t',header=None)
    # count the number of each cluster; cluster label is in the 7th column
    df_count = df[6].value_counts()
    # sorted by cluster label and save to df_all
    df_unique[cond] = list(df_count.sort_index())
# calculate the percentage of each cluster
df_unique['control'] = df_unique['control']/df_unique['control'].sum()*100
df_unique['degron'] = df_unique['degron']/df_unique['degron'].sum()*100
# plot the barplot
fig, ax = plt.subplots()
df_unique.plot(kind='bar',ax=ax)
plt.ylim(0, 34)
plt.xlabel('Cluster')
plt.ylabel('Percentage (%)')
# make x labels 1-6
#plt.xticks(range(6),range(1,7),rotation='vertical')
plt.title('Percentage of unique loops')
plt.savefig(f'{ol}barplot_unique_percentage.pdf')  

# plot the difference
df_diff = df_unique - df_all
fig, ax = plt.subplots()
df_diff.plot(kind='bar',ax=ax)
plt.xlabel('Cluster')
plt.ylabel('Percentage (%)')
# make x labels 1-6
#plt.xticks(range(6),range(1,7))
plt.title('Difference of percentage between unique and all loops')
plt.savefig(f'{ol}barplot_diff_percentage.pdf')
