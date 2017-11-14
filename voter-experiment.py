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
voter_data.loc[
    voter_data.eval('{}.notnull() | {}.notnull() | {}.notnull()'.format(
        *vars['ovp_vignette'])), 'treatment_groups'] = 'ÖVP vignette'
voter_data.loc[
    voter_data.eval('{}.notnull() | {}.notnull() | {}.notnull()'.format(
        *vars['spo_vignette'])), 'treatment_groups'] = 'SPÖ vignette'

# create identifier for partisan groups
voter_data['partisan_id'] = voter_data['w1_q19']
partisan_dict = {1: 'SPÖ partisan', 2: 'ÖVP partisan'}
partisan_dict.update({i: 'Non-partisan/other' for i in range(3, 8)})
voter_data.loc[:, 'partisan_id'] = voter_data['partisan_id'].replace(
    partisan_dict)


##########################
# descriptive statistics #
##########################

# number of non-missing observations in variables
for x in vars.values():
    print(voter_data[x].count())

# mean differences: SPÖ position (comparing SPÖ/ÖVP vignettes)
spo_mean_vig_ovp = voter_data[spo_pos_vig_ovp].mean()
spo_mean_vig_spo = voter_data[spo_pos_vig_spo].mean()
print(stats.ttest_ind(voter_data[spo_pos_vig_ovp], voter_data[spo_pos_vig_spo],
                      equal_var=False, nan_policy='omit'))

sns.violinplot(x="treatment_groups", y="spo_pos", data=voter_data)
plt.xlabel('')
plt.ylabel('SPÖ placement on asylum law')
plt.yticks(np.arange(0, 11, 2),
           (['soften' if x == 0 else
             'tighten' if x == 10 else
             x for x in np.arange(0, 11, 2)]))
plt.plot([0, 1], [spo_mean_vig_ovp, spo_mean_vig_spo],
         linestyle='-', marker='o', color='#e87e7e')
plt.show()

# mean differences: ÖVP position (comparing SPÖ/ÖVP vignettes)
ovp_mean_vig_ovp = voter_data[ovp_pos_vig_ovp].mean()
ovp_mean_vig_spo = voter_data[ovp_pos_vig_spo].mean()
print(stats.ttest_ind(voter_data[ovp_pos_vig_ovp], voter_data[ovp_pos_vig_spo],
                      equal_var=False, nan_policy='omit'))

sns.violinplot(x="treatment_groups", y="ovp_pos", data=voter_data)
plt.xlabel('')
plt.ylabel('ÖVP placement on asylum law')
plt.yticks(np.arange(0, 11, 2),
           (['soften' if x == 0 else
             'tighten' if x == 10 else
             x for x in np.arange(0, 11, 2)]))
plt.plot([0, 1], [ovp_mean_vig_ovp, ovp_mean_vig_spo],
         linestyle='-', marker='o', color='#e87e7e')
plt.show()

# stacked barplot for assessment of vignette
crosstab_spo = pd.crosstab(voter_data[assessment_vig_spo], voter_data['partisan_id'], margins=True)
crosstab_ovp = pd.crosstab(voter_data[assessment_vig_ovp], voter_data['partisan_id'], margins=True)

y = {}
ind = [0, 1, 3, 4, 6, 7]
for i in range(5):
    spov = crosstab_spo.iloc[i, :3] / crosstab_spo.iloc[5, :3]
    ovpv = crosstab_ovp.iloc[i, :3] / crosstab_ovp.iloc[5, :3]
    y[i] = sum(map(list, zip(spov, ovpv)), [])

plt.bar(ind, y[0])
plt.bar(ind, y[1], bottom=y[0])
bottom = [i + j for i, j in zip(y[0], y[1])]
plt.bar(ind, y[2], bottom=bottom)
bottom = [i + j for i, j in zip(bottom, y[2])]
plt.bar(ind, y[3], bottom=bottom)
bottom = [i + j for i, j in zip(bottom, y[3])]
plt.bar(ind, y[4], bottom=bottom)

plt.xticks([.5, 3.5, 6.5], ['Non-partisan/other',
                            'SPÖ partisan', 'ÖVP partisan'])
plt.xlabel('')
plt.ylabel('Assessment of asylum law')
plt.show()

sns.violinplot(x='partisan_id', y='assessment_vig',
               hue='treatment_groups', data=voter_data)
