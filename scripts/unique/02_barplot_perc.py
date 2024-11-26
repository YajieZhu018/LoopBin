# plot the percentage of each cluster in a barplot
import pandas as pd
import matplotlib.pyplot as plt
import os
# plot all loops
# input folder
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/intersect/01_sorted/'
# output folder 
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/02_plot/'
# make output folder
os.system(f'mkdir -p {ol}')
# create a dataframe to store the percentage of each cluster
df_all = pd.DataFrame()
# column names are control and degron
df_all['control'] = [0.0]*6
df_all['degron'] = [0.0]*6
# get the line number of the input file
# loop through control degron
for j in ['control','degron']:
    # loop through 1-6
    for i in range(1,7):
        file = f'{il}{j}_loops_label_{str(i)}.bed'
        # get the line number of the input file
        n = sum(1 for line in open(file))
        # save the line number to the dataframe
        df_all.loc[i-1,j] = n
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
plt.xticks(range(6),range(1,7),rotation='vertical')
plt.title('Percentage of all loops')
plt.savefig(f'{ol}barplot_all_percentage.pdf')  

# plot unique loops
# input folder
il = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/01_sorted/'
# output folder 
ol = '/usr/users/yzhu1/LoopBin/trials/saved_models/vade_7clusters_merged_control_degron_rep1/unique/02_plot/'
# create a dataframe to store the percentage of each cluster
df_unique = pd.DataFrame()
# column names are control and degron
df_unique['control'] = [0.0]*6
df_unique['degron'] = [0.0]*6
# get the line number of the input file
# loop through control degron
for j in ['control','degron']:
    # loop through 1-6
    for i in range(1,7):
        file = f'{il}{j}_loops_label_{str(i)}.bed'
        # get the line number of the input file
        n = sum(1 for line in open(file))
        # save the line number to the dataframe
        df_unique.loc[i-1,j] = n
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
plt.xticks(range(6),range(1,7),rotation='vertical')
plt.title('Percentage of unique loops')
plt.savefig(f'{ol}barplot_unique_percentage.pdf')  

# plot the difference
df_diff = df_unique - df_all
fig, ax = plt.subplots()
df_diff.plot(kind='bar',ax=ax)
plt.xlabel('Cluster')
plt.ylabel('Percentage (%)')
# make x labels 1-6
plt.xticks(range(6),range(1,7))
plt.title('Difference of percentage between unique and all loops')
plt.savefig(f'{ol}barplot_diff_percentage.pdf')
