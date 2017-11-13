import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

sns.set(style="whitegrid", palette="pastel", color_codes=True)

# path to Stata file
stata_file = 'data/AUTNES_OPS_2017_w1-4_DE.dta'

# relevant variables
spo_pos_pre = 'w2_q23x1'       # perceived SPÖ pos. on asylum law in 2nd wave
ovp_pos_pre = 'w2_q23x2'       # perceived ÖVP pos. on asylum law in 2nd wave
spo_pos_vig_ovp = 'w3_q37x1'   # SPÖ position after vignette with ÖVP label
ovp_pos_vig_ovp = 'w3_q37x2'   # ÖVP position after vignette with ÖVP label
spo_pos_vig_spo = 'w3_q39x1'   # SPÖ position after vignette with SPÖ label
ovp_pos_vig_spo = 'w3_q39x2'   # ÖVP position after vignette with SPÖ label
assessment_vig_ovp = 'w3_q36'       # assessment of the bill after ÖVP vignette
assessment_vig_spo = 'w3_q38'       # assessment of the bill after SPÖ vignette

vars = {'pre': [spo_pos_pre, ovp_pos_pre],
        'ovp_vignette': [spo_pos_vig_ovp, ovp_pos_vig_ovp, assessment_vig_ovp],
        'spo_vignette': [spo_pos_vig_spo, ovp_pos_vig_spo, assessment_vig_spo]}

# read Stata file
voter_data = pd.read_stata(stata_file, convert_categoricals=False)

# create stacked post-vignette variables by filling NAs in treatment groups
voter_data['spo_pos'] = voter_data[spo_pos_vig_ovp].fillna(
    voter_data[spo_pos_vig_spo])
voter_data['ovp_pos'] = voter_data[ovp_pos_vig_ovp].fillna(
    voter_data[ovp_pos_vig_spo])
voter_data['assessment_vig'] = voter_data[assessment_vig_ovp].fillna(
    voter_data[assessment_vig_spo])

# create identifier for treatment groups (TODO: more elegant solution?)
voter_data['treatment_groups'] = np.nan
voter_data.ix[voter_data.eval('{}.notnull() | {}.notnull() | {}.notnull()'.format(
    *vars['ovp_vignette'])), 'treatment_groups'] = 'ÖVP vignette'
voter_data.ix[voter_data.eval('{}.notnull() | {}.notnull() | {}.notnull()'.format(
    *vars['spo_vignette'])), 'treatment_groups'] = 'SPÖ vignette'


##############################
### descriptive statistics ###
##############################

# number of non-missing observations in variables
for x in vars.values():
    print(voter_data[x].count())

# mean differences: SPÖ position (comparing SPÖ/ÖVP vignettes)
spo_mean_vig_ovp = voter_data[spo_pos_vig_ovp].mean()
spo_mean_vig_spo = voter_data[spo_pos_vig_spo].mean()
print(stats.ttest_ind(voter_data[spo_pos_vig_ovp], voter_data[spo_pos_vig_spo],
                      equal_var=False, nan_policy='omit'))

spo_plot1 = sns.violinplot(x="treatment_groups", y="spo_pos", data=voter_data)
plt.plot([0, 1], [spo_mean_vig_ovp, spo_mean_vig_spo],
         linestyle='-', marker='o', color='#e87e7e')
spo_plot1.set(xlabel='', ylabel='SPÖ placement on asylum law')
plt.yticks(np.arange(0, 11, 2),
           (['soften' if x == 0 else
             'tighten' if x == 10 else
             x for x in np.arange(0, 11, 2)]))
plt.show()

# mean differences: ÖVP position (comparing SPÖ/ÖVP vignettes)
ovp_mean_vig_ovp = voter_data[ovp_pos_vig_ovp].mean()
ovp_mean_vig_spo = voter_data[ovp_pos_vig_spo].mean()
print(stats.ttest_ind(voter_data[ovp_pos_vig_ovp], voter_data[ovp_pos_vig_spo],
                      equal_var=False, nan_policy='omit'))

ovp_plot1 = sns.violinplot(x="treatment_groups", y="ovp_pos", data=voter_data)
plt.plot([0, 1], [ovp_mean_vig_ovp, ovp_mean_vig_spo],
         linestyle='-', marker='o', color='#e87e7e')
ovp_plot1.set(xlabel='', ylabel='ÖVP placement on asylum law')
plt.yticks(np.arange(0, 11, 2),
           (['soften' if x == 0 else
             'tighten' if x == 10 else
             x for x in np.arange(0, 11, 2)]))
plt.show()
