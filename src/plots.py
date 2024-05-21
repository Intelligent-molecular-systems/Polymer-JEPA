import numpy as np
import matplotlib.pyplot as plt

### EA DATA SCENARIOS ###
# Data from the table for the Aldeghi dataset performance for EA
# ft_size = [0.4, 0.8, 1.6, 2.4, 3.2, 4]  # Finetune sizes in % # , 8, 16, 24
# r2_no_prtrn_ea = np.array([0.46, 0.71, 0.83, 0.87, 0.92, 0.94]) # , 0.96, 0.98, 0.99
# r2_prtrn_ea = np.array([0.67, 0.76, 0.86, 0.88, 0.90, 0.93]) # , 0.97, 0.98, 0.99

# std_dev_no_prtrn_ea = np.array([0.15, 0.06, 0.05, 0.04, 0.01, 0.01]) # , 0.002, 0.004, 0.002
# std_dev_prtrn_ea = np.array([0.01, 0.01, 0.02, 0.02, 0.01, 0.005]) # , 0.0001, 0.0005, 0.0001

# # Create plot for Aldeghi dataset performance for EA
# plt.figure(figsize=(10, 6))
# plt.title("Pretraining Performance on Aldeghi Dataset for EA")
# plt.xlabel("Finetune dataset size (%)")
# # plt.ylabel(r'$R^2$')

# # Plotting lines with standard deviation areas

# plt.plot(ft_size, r2_no_prtrn_ea, label='No Pretraining', color='blue')
# plt.fill_between(ft_size, r2_no_prtrn_ea - std_dev_no_prtrn_ea, r2_no_prtrn_ea + std_dev_no_prtrn_ea, color='blue', alpha=0.05)

# plt.plot(ft_size, r2_prtrn_ea, label='Pretrained', color='green')
# plt.fill_between(ft_size, r2_prtrn_ea - std_dev_prtrn_ea, r2_prtrn_ea + std_dev_prtrn_ea, color='green', alpha=0.05)
# plt.xticks(ft_size, [f"{size}%" for size in ft_size])  # Converts size to percentage strings

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()


# # ### IP DATA SCENARIOS ###
# # Data from the new table
# ft_size = [0.4, 0.8, 1.6, 2.4, 3.2, 4]  # Finetune sizes in % #, 8, 16, 24
# r2_no_prtrn = np.array([0.57, 0.75, 0.86, 0.91, 0.94, 0.95]) # , 0.97, 0.99, 0.99
# r2_prtrn = np.array([0.68, 0.81, 0.88, 0.92, 0.94, 0.95]) #, 0.97, 0.99, 0.99

# std_dev_no_prtrn_aldeghi = np.array([0.08, 0.03, 0.02, 0.02, 0.003, 0.009]) # , 0.003, 0.003, 0.0005
# std_dev_prtrn_aldeghi = np.array([0.04, 0.03, 0.01, 0.01, 0.01, 0.01]) # , 0.005, 0.0005, 0.0001

# # Create plot for Aldeghi dataset
# plt.figure(figsize=(10, 6))
# plt.title("Pretraining Performance on Aldeghi Dataset for IP")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel(r'$R^2$')

# # Plotting lines with standard deviation areas
# plt.plot(ft_size, r2_no_prtrn, label='No Pretraining', color='blue')
# plt.fill_between(ft_size, r2_no_prtrn - std_dev_no_prtrn_aldeghi, r2_no_prtrn + std_dev_no_prtrn_aldeghi, color='blue', alpha=0.05)

# plt.plot(ft_size, r2_prtrn, label='Pretrained', color='green')
# plt.fill_between(ft_size, r2_prtrn - std_dev_prtrn_aldeghi, r2_prtrn + std_dev_prtrn_aldeghi, color='green', alpha=0.05)
# plt.xticks(ft_size, [f"{size}%" for size in ft_size])  # Converts size to percentage strings

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()


### DIBLOCK DATA SCENARIOS ###

# Data from the table
ft_size = [4, 8, 12, 16, 24, 32, 48, 80]
auprc_no_prtrn = np.array([0.36, 0.44, 0.50, 0.53, 0.60, 0.65, 0.68, 0.71])
auprc_prtrn = np.array([0.40, 0.50, 0.57, 0.61, 0.65, 0.67, 0.70, 0.72])

std_dev_no_prtrn = np.array([0.03, 0.01, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02])
std_dev_prtrn = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.03])


# Create plot again with the updated x-axis label
plt.figure(figsize=(10, 6))
plt.title("Pretraining Performance on Diblock Dataset")
plt.xlabel("Finetune dataset size (%)")  # Updated label
plt.ylabel("AUPRC")

# Plotting lines with standard deviation areas again

plt.plot(ft_size, auprc_no_prtrn, label='No Pretraining', color='blue')
plt.fill_between(ft_size, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.1)

plt.plot(ft_size, auprc_prtrn, label='Pretraining', color='green')
plt.fill_between(ft_size, auprc_prtrn - std_dev_prtrn, auprc_prtrn + std_dev_prtrn, color='green', alpha=0.1)
plt.xticks(ft_size, [f"{size}%" for size in ft_size])  # Converts size to percentage strings

# Add legend and show plot
plt.legend()
plt.grid(True)
plt.show()


## ALDEGHI EA VS RANDOM FOREST IN SMALL DATASET ###
# 0,818721716073818	0,85656229569584	0,921687052812763	0,950066061883963	0,964727926173172	0,973019307267574
# ft_size_aldeghi_comparison = [0.4, 0.8, 1.6, 4]  # Finetune sizes in %
# r2_wdmpnn_prtrn = np.array([0.67, 0.76, 0.86, 0.93])
# r2_rf_no_prtrn = np.array([0.87, 0.87, 0.88, 0.89])
# r2_no_prtrn_ea = np.array([0.46, 0.71, 0.83, 0.94]) # , 0.96, 0.98, 0.99
# r2_gao_prtrn_ = np.array([0.82, 0.86, 0.92, 0.99])
# r2_gao_baseline = np.array([0.63, 0.69,	0.80, 0.95])
# r2_gao_prtrn_only_encoder = np.array([0.71, 0.81, 0.89, 0.96])

# std_dev_wdmpnn_prtrn = np.array([0.01, 0.01, 0.02, 0.005])
# std_dev_rf_no_prtrn = np.array([0.02, 0.02, 0.02, 0.02])
# std_dev_no_prtrn_ea = np.array([0.15, 0.06, 0.05, 0.01]) # , 0.002, 0.004, 0.002
# std_dev_gao_prtrn = np.array([0.02, 0.02, 0.01, 0.005])
# std_dev_gao_baseline = np.array([0.02, 0.03, 0.02, 0.01])
# std_dev_gao_prtrn_only_encoder = np.array([0.02, 0.01, 0.01, 0.01])

# # Create plot for the comparison between Pretrained WDMPNN and Random Forest not pretrained
# plt.figure(figsize=(10, 6))
# # plt.title("Pretrained WDMPNN vs Random Forest on Aldeghi Dataset")
# plt.title("Pretrained WDMPNN vs non-pretrained WDMPNN on the Aldeghi and Coley Dataset")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel(r"$R^2$")

# # Plotting lines with standard deviation areas
# plt.plot(ft_size_aldeghi_comparison, r2_wdmpnn_prtrn, label='WDMPNN - Pretrained', color='green')
# plt.fill_between(ft_size_aldeghi_comparison, r2_wdmpnn_prtrn - std_dev_wdmpnn_prtrn, r2_wdmpnn_prtrn + std_dev_wdmpnn_prtrn, color='green', alpha=0.05)

# # plt.plot(ft_size_aldeghi_comparison, r2_rf_no_prtrn, label='Random Forest - No pretraining', color='red')
# # plt.fill_between(ft_size_aldeghi_comparison, r2_rf_no_prtrn - std_dev_rf_no_prtrn, r2_rf_no_prtrn + std_dev_rf_no_prtrn, color='red', alpha=0.05)

# plt.plot(ft_size_aldeghi_comparison, r2_no_prtrn_ea, label='WDMPNN - No Pretraining', color='blue')
# plt.fill_between(ft_size_aldeghi_comparison, r2_no_prtrn_ea - std_dev_no_prtrn_ea, r2_no_prtrn_ea + std_dev_no_prtrn_ea, color='blue', alpha=0.05)

# # plt.plot(ft_size_aldeghi_comparison, r2_gao_prtrn_, label='Other SSL tasks - All layers transfer', color='orange')
# # plt.fill_between(ft_size_aldeghi_comparison, r2_gao_prtrn_ - std_dev_gao_prtrn, r2_gao_prtrn_ + std_dev_gao_prtrn, color='orange', alpha=0.05)

# # plt.plot(ft_size_aldeghi_comparison, r2_gao_baseline, label='Gao Baseline', color='purple')
# # plt.fill_between(ft_size_aldeghi_comparison, r2_gao_baseline - std_dev_gao_baseline, r2_gao_baseline + std_dev_gao_baseline, color='purple', alpha=0.05)

# # plt.plot(ft_size_aldeghi_comparison, r2_gao_prtrn_only_encoder, label='Other SSL tasks - Only encoder layers transfer', color='grey')
# # plt.fill_between(ft_size_aldeghi_comparison, r2_gao_prtrn_only_encoder - std_dev_gao_prtrn_only_encoder, r2_gao_prtrn_only_encoder + std_dev_gao_prtrn_only_encoder, color='grey', alpha=0.05)

# # Adding custom x-axis markers
# plt.xticks(ft_size_aldeghi_comparison, [f"{size}%" for size in ft_size_aldeghi_comparison])

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()

# ### DIBLOCK VS RANDOM FOREST IN SMALL DATASET ###
# ft_size_diblock = [4, 24, 48, 80]  # Feature set sizes in %
# auprc_wdmpnn_prtrn = np.array([0.40, 0.65, 0.70, 0.72])
# auprc_rf_no_prtrn = np.array([0.59, 0.70, 0.71, 0.74])
# auprc_no_prtrn = np.array([0.36, 0.60, 0.68, 0.71])

# std_dev_wdmpnn_prtrn = np.array([0.02, 0.02, 0.01, 0.03])
# std_dev_rf_no_prtrn = np.array([0.01, 0.01, 0.01, 0.01])
# std_dev_no_prtrn = np.array([0.03, 0.02, 0.02, 0.02])

# # Create plot for the comparison between Pretrained WDMPNN and Random Forest not pretrained on Diblock dataset
# plt.figure(figsize=(10, 6))
# plt.title("Pretrained WDMPNN vs non-pretrained WDMPNN on Diblock Dataset")
# plt.xlabel("Finetune dataset size (%)")
# plt.ylabel("AUPRC")

# # Plotting lines with standard deviation areas
# plt.plot(ft_size_diblock, auprc_wdmpnn_prtrn, label='WDMPNN - Pretrained', color='green')
# plt.fill_between(ft_size_diblock, auprc_wdmpnn_prtrn - std_dev_wdmpnn_prtrn, auprc_wdmpnn_prtrn + std_dev_wdmpnn_prtrn, color='green', alpha=0.05)

# # plt.plot(ft_size_diblock, auprc_rf_no_prtrn, label='RF - No Pretraining', color='red')
# # plt.fill_between(ft_size_diblock, auprc_rf_no_prtrn - std_dev_rf_no_prtrn, auprc_rf_no_prtrn + std_dev_rf_no_prtrn, color='red', alpha=0.05)

# plt.plot(ft_size_diblock, auprc_no_prtrn, label='WDMPNN - No Pretraining', color='blue')
# plt.fill_between(ft_size_diblock, auprc_no_prtrn - std_dev_no_prtrn, auprc_no_prtrn + std_dev_no_prtrn, color='blue', alpha=0.1)


# # Adding custom x-axis markers
# plt.xticks(ft_size_diblock, [f"{size}%" for size in ft_size_diblock])

# # Add legend and show plot
# plt.legend()
# plt.grid(True)
# plt.show()

