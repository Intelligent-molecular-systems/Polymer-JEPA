import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### DIBLOCK DATA SCENARIOS ###
plt.figure(0, figsize=(8, 5))

# Original result data from the table
ft_size = [4, 8, 12, 16, 24, 32, 48, 80]
auprc_no_prtrn = np.array([0.36, 0.44, 0.50, 0.53, 0.60, 0.65, 0.68, 0.71])
auprc_prtrn = np.array([0.40, 0.50, 0.57, 0.61, 0.65, 0.67, 0.70, 0.72])

std_dev_no_prtrn = np.array([0.03, 0.01, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02])
std_dev_prtrn = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.03])

plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel("AUPRC", fontsize=20)

# # Plotting lines with standard deviation areas again

plt.plot(ft_size, auprc_prtrn, label='Jepa (wD-MPNN) - pretrained', color='green')
plt.fill_between(ft_size, auprc_prtrn - std_dev_prtrn, auprc_prtrn + std_dev_prtrn, color='green', alpha=0.05)
plt.plot(ft_size, auprc_no_prtrn, label='wD-MPNN - no pretraining', color='blue')
plt.fill_between(ft_size, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.05)

plt.xticks(ft_size, [f"{size}%" for size in ft_size], rotation=45, fontsize=18)  # Converts size to percentage strings
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('Results/experiments_paper/diblock_comparison.png', dpi=300, bbox_inches='tight')


# Updated values after cross validation
""" ft_size_aldeghi_comparison = [0.4, 0.8, 1.6, 4]  # Finetune sizes in %
r2_wDMPNN_only_encoder_prtrn = np.array([0.67, 0.76, 0.86, 0.93])
r2_wDMPNN_with_mw_prtrn = np.array([0.73, 0.82, 0.87, 0.94])
r2_rf_no_prtrn = np.array([0.87, 0.87, 0.88, 0.89])
# r2_no_prtrn_ea = np.array([0.46, 0.71, 0.83, 0.94]) # , 0.96, 0.98, 0.99
r2_gao_prtrn_ = np.array([0.695654, 0.763982, 0.852779, 0.954708])
# r2_gao_baseline = np.array([0.63, 0.69,	0.80, 0.95])
r2_gao_prtrn_only_encoder = np.array([0.636246, 0.741461, 0.839005, 0.944453])

std_dev_wDMPNN_only_encoder_prtrn = np.array([0.01, 0.01, 0.02, 0.005])
std_dev_wDMPNN_with_mw_prtrn = np.array([0.03, 0.01, 0.03, 0.01])
std_dev_rf_no_prtrn = np.array([0.02, 0.02, 0.02, 0.02])
# std_dev_no_prtrn_ea = np.array([0.15, 0.06, 0.05, 0.01]) # , 0.002, 0.004, 0.002
std_dev_gao_prtrn = np.array([0.039554, 0.033191, 0.030189, 0.007161])
# std_dev_gao_baseline = np.array([0.02, 0.03, 0.02, 0.01])
std_dev_gao_prtrn_only_encoder = np.array([0.052568, 0.048467, 0.025704, 0.020673])

# Create plot for the comparison between Pretrained wD-MPNN and Random Forest not pretrained
plt.figure(2, figsize=(10, 6))
# plt.title("Pretrained wD-MPNN vs Random Forest on Aldeghi Dataset")
#plt.title("Our pretraining vs other SSL tasks on Aldeghi Dataset")
plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)

# Plotting lines with standard deviation areas
plt.plot(ft_size_aldeghi_comparison, r2_wDMPNN_only_encoder_prtrn, label='JEPA', color='green') # - Only encoder layer transfer
plt.fill_between(ft_size_aldeghi_comparison, r2_wDMPNN_only_encoder_prtrn - std_dev_wDMPNN_only_encoder_prtrn, r2_wDMPNN_only_encoder_prtrn + std_dev_wDMPNN_only_encoder_prtrn, color='green', alpha=0.05)

plt.plot(ft_size_aldeghi_comparison, r2_wDMPNN_with_mw_prtrn, label='JEPA & pseudolabel - All layers transfer', color='purple')
plt.fill_between(ft_size_aldeghi_comparison, r2_wDMPNN_with_mw_prtrn - std_dev_wDMPNN_with_mw_prtrn, r2_wDMPNN_with_mw_prtrn + std_dev_wDMPNN_with_mw_prtrn, color='purple', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_rf_no_prtrn, label='Random Forest - No pretraining', color='red')
# plt.fill_between(ft_size_aldeghi_comparison, r2_rf_no_prtrn - std_dev_rf_no_prtrn, r2_rf_no_prtrn + std_dev_rf_no_prtrn, color='red', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_no_prtrn_ea, label='wD-MPNN - No Pretraining', color='blue')
# plt.fill_between(ft_size_aldeghi_comparison, r2_no_prtrn_ea - std_dev_no_prtrn_ea, r2_no_prtrn_ea + std_dev_no_prtrn_ea, color='blue', alpha=0.05)

plt.plot(ft_size_aldeghi_comparison, r2_gao_prtrn_, label='Other SSL tasks - All layers transfer', color='orange')
plt.fill_between(ft_size_aldeghi_comparison, r2_gao_prtrn_ - std_dev_gao_prtrn, r2_gao_prtrn_ + std_dev_gao_prtrn, color='orange', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_gao_baseline, label='Gao Baseline', color='purple')
# plt.fill_between(ft_size_aldeghi_comparison, r2_gao_baseline - std_dev_gao_baseline, r2_gao_baseline + std_dev_gao_baseline, color='purple', alpha=0.05)

plt.plot(ft_size_aldeghi_comparison, r2_gao_prtrn_only_encoder, label='Other SSL tasks', color='grey') #  - Only encoder layers transfer
plt.fill_between(ft_size_aldeghi_comparison, r2_gao_prtrn_only_encoder - std_dev_gao_prtrn_only_encoder, r2_gao_prtrn_only_encoder + std_dev_gao_prtrn_only_encoder, color='grey', alpha=0.05)

# Adding custom x-axis markers
plt.xticks(ft_size_aldeghi_comparison, [f"{size}%" for size in ft_size_aldeghi_comparison])

# Add legend and show plot
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('aldeghi_comparison.png', dpi=300, bbox_inches='tight') """

# ### DIBLOCK VS RANDOM FOREST IN SMALL DATASET ###
# ft_size_diblock = [4, 24, 48, 80]  # Feature set sizes in %
# auprc_wD-MPNN_prtrn = np.array([0.40, 0.65, 0.70, 0.72])
# auprc_rf_no_prtrn = np.array([0.59, 0.70, 0.71, 0.74])
# auprc_no_prtrn = np.array([0.36, 0.60, 0.68, 0.71])

# std_dev_wD-MPNN_prtrn = np.array([0.02, 0.02, 0.01, 0.03])
# std_dev_rf_no_prtrn = np.array([0.01, 0.01, 0.01, 0.01])
# std_dev_no_prtrn = np.array([0.03, 0.02, 0.02, 0.02])

# # Create plot for the comparison between Pretrained wD-MPNN and Random Forest not pretrained on Diblock dataset
# plt.figure(figsize=(10, 6))
# plt.title("Pretrained wD-MPNN vs non-pretrained wD-MPNN on Diblock Dataset")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel("AUPRC")

# # Plotting lines with standard deviation areas
# plt.plot(ft_size_diblock, auprc_wD-MPNN_prtrn, label='wD-MPNN - Pretrained', color='green')
# plt.fill_between(ft_size_diblock, auprc_wD-MPNN_prtrn - std_dev_wD-MPNN_prtrn, auprc_wD-MPNN_prtrn + std_dev_wD-MPNN_prtrn, color='green', alpha=0.05)

# # plt.plot(ft_size_diblock, auprc_rf_no_prtrn, label='RF - No Pretraining', color='red')
# # plt.fill_between(ft_size_diblock, auprc_rf_no_prtrn - std_dev_rf_no_prtrn, auprc_rf_no_prtrn + std_dev_rf_no_prtrn, color='red', alpha=0.05)

# plt.plot(ft_size_diblock, auprc_no_prtrn, label='wD-MPNN - No Pretraining', color='blue')
# plt.fill_between(ft_size_diblock, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.1)


# # Adding custom x-axis markers
# plt.xticks(ft_size_diblock, [f"{size}%" for size in ft_size_diblock])

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()

# ### FROZEN ENCODER WEIGHTS WHEN FINETUNING - ALDEGHI EA ### 
# ft_size_aldeghi_comparison = [0.4, 0.8, 1.6, 4, 8]  # Finetune sizes in % of full dataset # , 16
# r2_pretrained = np.array([0.016, 0.1747, 0.3623, 0.4802, 0.5621]) #  0.641
# r2_no_pretrained = np.array([-0.0001, 0.1071, 0.4756, 0.73, 0.84]) # , 0.88


# std_dev_pretrained = np.array([0.04, 0.04, 0.03, 0.03, 0.01]) # , 0.02
# std_dev_no_pretrained = np.array([0.02, 0.05, 0.08, 0.01, 0.01]) # , 0.01

# # Create plot for the comparison between Pretrained wD-MPNN and Random Forest not pretrained
# plt.figure(figsize=(10, 6))
# # plt.title("Pretrained wD-MPNN vs Random Forest on Aldeghi Dataset")
# plt.title("Pretraning vs No pretraining on Aldeghi Dataset with frozen encoder weights - EA property")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel(r"$R^2$")

# # Plotting lines with standard deviation areas
# plt.plot(ft_size_aldeghi_comparison, r2_pretrained, label='JEPA - Pretrained', color='green')
# plt.fill_between(ft_size_aldeghi_comparison, r2_pretrained - std_dev_pretrained, r2_pretrained + std_dev_pretrained, color='green', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_no_pretrained, label='JEPA - Not pretrained', color='purple')
# plt.fill_between(ft_size_aldeghi_comparison, r2_no_pretrained - std_dev_no_pretrained, r2_no_pretrained + std_dev_no_pretrained, color='purple', alpha=0.05)

# # Adding custom x-axis markers
# plt.xticks(ft_size_aldeghi_comparison, [f"{size}%" for size in ft_size_aldeghi_comparison])

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()

""" New plots with updated experiments from csv files. """
# Load the CSV
df = pd.read_csv('Results/experiments_paper/summary_statistics.csv', sep=';')  # Replace with your actual filename
df_other_paper = pd.read_csv('Results/experiments_paper/summary_statistics_Gao.csv')

# Convert 'TRUE'/'FALSE' strings to booleans if needed
df['PL'] = df['PL'].astype(bool)
df['pretraining'] = df['pretraining'].astype(bool)

# Grouping data by percentage and a desired configuration (e.g. PL=True, pretraining=True)
# Different subsets
plt.figure(3,figsize=(8, 5))
df_PL_x_PT_0_N_0 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['pretraining'] == False) & (df['norm'] == 0)].sort_values(by='percentage')
df_PL_x_PT_0_N_1 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.1) & (df['pretraining'] == False) & (df['norm'] == 1)].sort_values(by='percentage')
df_PL_1_PT_1_N_0 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['PL'] == True) & (df['pretraining'] == True) & (df['norm'] == 0)].sort_values(by='percentage')
df_PL_1_PT_1_N_1 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.1) & (df['PL'] == True) & (df['pretraining'] == True) & (df['norm'] == 1)].sort_values(by='percentage')
df_PL_0_PT_1_N_0 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.2) & (df['PL'] == False) & (df['pretraining'] == True) & (df['norm'] == 0)].sort_values(by='percentage')
df_PL_0_PT_1_N_1 = df[(df['percentage'] >= 0.01) & (df['percentage'] <= 0.1) & (df['PL'] == False) & (df['pretraining'] == True) & (df['norm'] == 1)].sort_values(by='percentage')

# Match subsets with desired labels and colors
subsets = [df_PL_1_PT_1_N_0, df_PL_0_PT_1_N_0]
#labels_colors = [("Jepa - pretrained","green"), ("Jepa - no pretraining (Baseline)","blue")]#, ("Jepa & pseudolabel - All layers transferred", "purple")]
labels_colors = [("Jepa - PL","green"), ("Jepa - No PL", "green")]# ("Jepa - No pretraining", "blue"), ("Jepa - Pretrained", "orange"), ("Jepa - Only encoder layers transferred", "grey") ]
#subsets = [df_PL_x_PT_0_N_0,df_PL_x_PT_0_N_1,df_PL_1_PT_1_N_0,df_PL_1_PT_1_N_1, df_PL_0_PT_1_N_0, df_PL_0_PT_1_N_1]

for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['R2_mean']
    y_std = exp_sub['R2_std']
    #label = 'Polymer-JEPA (PL='+str(exp_sub["PL"].iloc[0])+', PT='+str(exp_sub["pretraining"].iloc[0])+')'
    #label = str(exp_sub["PL"].iloc[0])+str(exp_sub["pretraining"].iloc[0])+str(exp_sub["norm"].iloc[0])
    # Plot with shaded standard deviation
    if "No PL" in l_c[0]:
        plt.plot(x, y, label=l_c[0], color=l_c[1], linestyle='--')
    else: 
        plt.plot(x, y, label=l_c[0], color=l_c[1])

    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=l_c[1])


# Results of the other paper, Only_enc_transfer
df_no_SS = df_other_paper[df_other_paper['source'] == 'No_SS'].sort_values(by='percentage')
df_no_SS['source'] = df_no_SS['source'].replace({'No_SS':'baseline'})
df_only_enc_transfer = df_other_paper[df_other_paper['source'] == 'N-SSL'].sort_values(by='percentage')
df_all_layer_transfer = df_other_paper[df_other_paper['source'] == 'NG-SSL'].sort_values(by='percentage')

subsets_other_paper = [df_all_layer_transfer, df_only_enc_transfer]#, df_only_enc_transfer]#, df_only_enc_transfer, df_no_SS]#, df_only_enc_transfer, df_all_layer_transfer]
labels_colors_other_paper = [("Gao et al. - PL", "magenta"), ("Gao et al. - No PL", "magenta")]#, ("Gao et al. - Baseline", "red")]
#labels_colors_other_paper = [("Gao et al. - NG-SSL (best)", "magenta")]


for exp_sub, l_c in zip(subsets_other_paper, labels_colors_other_paper):
    x = exp_sub['percentage']*40
    y = exp_sub['mean_R2']
    y_std = exp_sub['std_R2']
    #label = "Gao et al. ("+str(exp_sub["source"].iloc[0])+')'
    # Plot with shaded standard deviation
    if "No PL" in l_c[0]:
        plt.plot(x, y, label=l_c[0], color=l_c[1], linestyle = "--")
    else: 
        plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=l_c[1])


plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/comparison_JEPA_GaoBest_PL.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot all precentages of Gao et al. work 
plt.figure(4,figsize=(8, 5))
df_other_paper_all = pd.read_csv('Results/experiments_paper/summary_statistics_Gao_all_perc.csv')

df_no_SS = df_other_paper_all[(df_other_paper_all['percentage'] >= 0.01) & (df_other_paper_all['percentage'] <= 0.4) & (df_other_paper_all['source'] == 'No_SS')].sort_values(by='percentage')
df_only_enc_transfer = df_other_paper_all[(df_other_paper_all['percentage'] >= 0.01) & (df_other_paper_all['percentage'] <= 0.4) & (df_other_paper_all['source'] == 'N-SSL')].sort_values(by='percentage')
df_all_layer_transfer = df_other_paper_all[(df_other_paper_all['percentage'] >= 0.01) & (df_other_paper_all['percentage'] <= 0.4) & (df_other_paper_all['source'] == 'NG-SSL')].sort_values(by='percentage')

subsets_other_paper = [df_no_SS, df_only_enc_transfer, df_all_layer_transfer]

for exp_sub in subsets_other_paper:
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['mean_R2']
    y_std = exp_sub['std_R2']
    label = "Gao et al. ("+str(exp_sub["source"].iloc[0])+')'
    # Plot with shaded standard deviation

    plt.plot(x, y, label=label, linestyle='--')
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.1)


plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/Gao_all_perc_CV.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot only comparison of pretraining and no pretraining with pseudolabel false 
plt.figure(5,figsize=(8, 5))
subsets = [df_PL_0_PT_1_N_0, df_PL_x_PT_0_N_0]
labels_colors = [('JEPA (wD-MPNN) - pretrained',"green"), ("wD-MPNN - no pretraining", "blue")]
for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['R2_mean']
    y_std = exp_sub['R2_std']
    #label = 'Polymer-JEPA (Pretraining='+str(exp_sub["pretraining"].iloc[0])+')'
    #label = str(exp_sub["PL"].iloc[0])+str(exp_sub["pretraining"].iloc[0])+str(exp_sub["norm"].iloc[0])
    # Plot with shaded standard deviation
    
    plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.05, color=l_c[1])


plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=18)
plt.grid(True)
plt.savefig('Results/experiments_paper/Pretrain_nopretrain_aldeghi_EA.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot comparison of pretraining and no pretraining with pseudolabel false and the RF model 
plt.figure(6,figsize=(8, 5))
subsets = [df_PL_1_PT_1_N_0, df_PL_x_PT_0_N_0]
labels_colors = [("Jepa (wD-MPNN) - pretrained","green"), ("wD-MPNN - No pretraining", "blue")]
for exp_sub, l_c in zip(subsets, labels_colors): 
    # Extract x and y values
    x = exp_sub['percentage']*40
    y = exp_sub['R2_mean']
    y_std = exp_sub['R2_std']
    #label = 'Polymer-JEPA (Pretraining='+str(exp_sub["pretraining"].iloc[0])+')'
    #label = str(exp_sub["PL"].iloc[0])+str(exp_sub["pretraining"].iloc[0])+str(exp_sub["norm"].iloc[0])
    # Plot with shaded standard deviation
    
    plt.plot(x, y, label=l_c[0], color=l_c[1])
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.05, color=l_c[1])

df_RF = pd.read_csv("Results/experiments_paper/summary_RF_aldeghi.csv")  # Update with actual path
# Baseline line in red
x_RF = df_RF["percentage"] * 40
y_RF = df_RF["R2_mean"]
y_std_RF = df_RF["R2_std"]

plt.plot(x, y_RF, label="Random Forest - No pretraining", color="red", linestyle="-")
plt.fill_between(x_RF, y_RF - y_std_RF, y_RF + y_std_RF, alpha=0.05, color="red")

plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel(r"$R^2$", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
# Adding custom x-axis markers
plt.xticks(x, [f"{size}%" for size in x])

# Add legend and show plot
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/Pretrain_nopretrain_RF_aldeghi_EA.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(7,figsize=(8, 5))
# Data from previous experiments: 
ft_size = [4, 8, 12, 16, 24, 32, 48, 80]
auprc_no_prtrn = np.array([0.36, 0.44, 0.50, 0.53, 0.60, 0.65, 0.68, 0.71])
auprc_prtrn = np.array([0.40, 0.50, 0.57, 0.61, 0.65, 0.67, 0.70, 0.72])

std_dev_no_prtrn = np.array([0.03, 0.01, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02])
std_dev_prtrn = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.03])

# # Plotting lines with standard deviation areas again

plt.plot(ft_size, auprc_prtrn, label='JEPA (wD-MPNN) - pretrained', color='green')
plt.fill_between(ft_size, auprc_prtrn - std_dev_prtrn, auprc_prtrn + std_dev_prtrn, color='green', alpha=0.05)
plt.plot(ft_size, auprc_no_prtrn, label='wD-MPNN - No pretraining', color='blue')
plt.fill_between(ft_size, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.05)

df_RF = pd.read_csv("Results/experiments_paper/summary_RF_diblock.csv") 
# Baseline line in red
x_RF = df_RF["finetune_percentage"] * 100
y_RF = df_RF["prc_mean"]
y_std_RF = df_RF["prc_std"]

plt.plot(x_RF, y_RF, label="Random Forest - No pretraining", color="red", linestyle="-")
plt.fill_between(x_RF, y_RF - y_std_RF, y_RF + y_std_RF, alpha=0.05, color="red")
plt.xlabel("Finetune dataset size (%)", fontsize=20)
plt.ylabel("AUPRC", fontsize=20)

plt.xticks(ft_size, [f"{size}%" for size in ft_size], rotation=45, fontsize=18)  # Converts size to percentage strings
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig('Results/experiments_paper/diblock_comparison_RF.png', dpi=300, bbox_inches='tight')