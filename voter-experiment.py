import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.formula.api import ols
from simulation import simulate, sim_predict

# seaborn plot style settings
sns.set(style='darkgrid', color_codes=True)
sns.set_palette('pastel')
party_colors = sns.color_palette(["#5b728a", "#e74c3c"])

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
voter_data['spo_pos'] = voter_data[spo_pos_vig_ovp].fillna(voter_data[spo_pos_vig_spo])
voter_data['ovp_pos'] = voter_data[ovp_pos_vig_ovp].fillna(voter_data[ovp_pos_vig_spo])
voter_data['assessment_vig'] = -1 * voter_data[assessment_vig_ovp].fillna(
    voter_data[assessment_vig_spo]) + 6  # flip the assessment scale

# create identifier for treatment groups
voter_data['treatment_groups'] = np.nan
eval_ovp = '{}.notnull() | {}.notnull() | {}.notnull()'.format(*vars['ovp_vignette'])
eval_spo = '{}.notnull() | {}.notnull() | {}.notnull()'.format(*vars['spo_vignette'])
voter_data.loc[voter_data.eval(eval_ovp), 'treatment_groups'] = 'ÖVP vignette'
voter_data.loc[voter_data.eval(eval_spo), 'treatment_groups'] = 'SPÖ vignette'
voter_data['spo_treatment'] = pd.get_dummies(voter_data['treatment_groups'])['SPÖ vignette']
voter_data['ovp_treatment'] = pd.get_dummies(voter_data['treatment_groups'])['ÖVP vignette']

# create identifier for partisan groups
voter_data['partisan_id'] = voter_data['w1_q19']
partisan_dict = {1: 'SPÖ partisans', 2: 'ÖVP partisans'}
partisan_dict.update({i: 'Non-partisans/other' for i in range(3, 8)})
voter_data.loc[:, 'partisan_id'] = voter_data['partisan_id'].replace(partisan_dict)


##########################
# descriptive statistics #
##########################

# number of non-missing observations in variables
for x in vars.values():
    print(voter_data[x].count())


def mean_ci(series):
    """
    mean_ci takes in a pandas.core.series.Series and returns its mean,
    plus a tuple with the 95% confidence interval of the mean.
    """
    if len(series) > 0:
        mean = series.mean()
        se = stats.sem(series, nan_policy='omit')
    return mean, (mean - 1.96 * se, mean + 1.96 * se)


# mean differences: SPÖ position (comparing SPÖ/ÖVP vignettes)
spo_mean_vig_ovp, spo_ci_vig_ovp = mean_ci(voter_data[spo_pos_vig_ovp])
spo_mean_vig_spo, spo_ci_vig_spo = mean_ci(voter_data[spo_pos_vig_spo])
print(stats.ttest_ind(voter_data[spo_pos_vig_ovp], voter_data[spo_pos_vig_spo],
                      equal_var=False, nan_policy='omit'))

# mean differences: ÖVP position (comparing SPÖ/ÖVP vignettes)
ovp_mean_vig_ovp, ovp_ci_vig_ovp = mean_ci(voter_data[ovp_pos_vig_ovp])
ovp_mean_vig_spo, ovp_ci_vig_spo = mean_ci(voter_data[ovp_pos_vig_spo])
print(stats.ttest_ind(voter_data[ovp_pos_vig_ovp], voter_data[ovp_pos_vig_spo],
                      equal_var=False, nan_policy='omit'))

# graph differences in SPÖ and ÖVP placements by treatment groups
fig, axs = plt.subplots(ncols=2)
sns.violinplot(x="treatment_groups", y="spo_pos", data=voter_data,
               ax=axs[0], palette=party_colors)
sns.violinplot(x="treatment_groups", y="ovp_pos", data=voter_data,
               ax=axs[1], palette=party_colors)
axs[0].set_title('SPÖ')
axs[1].set_title('ÖVP')
axs[0].set_ylabel('Policy placement on asylum law')
axs[1].set_ylabel('')
axs[0].set_xlabel('')
axs[1].set_xlabel('')
plot_range = np.arange(0, 11, 2)
axs[0].set_yticks(list(plot_range))
axs[0].set_yticklabels(['soften' if x == 0 else
                        'tighten' if x == 10 else
                        x for x in plot_range])
axs[0].plot([0, 1], [spo_mean_vig_ovp, spo_mean_vig_spo], linestyle='-', color='#e87e7e')
axs[0].plot([0, 0], [spo_ci_vig_ovp[0], spo_ci_vig_ovp[1]], linestyle='-', color='#e87e7e')
axs[0].plot([1, 1], [spo_ci_vig_spo[0], spo_ci_vig_spo[1]], linestyle='-', color='#e87e7e')
axs[1].plot([0, 1], [ovp_mean_vig_ovp, ovp_mean_vig_spo], linestyle='-', color='#e87e7e')
axs[1].plot([0, 0], [ovp_ci_vig_ovp[0], ovp_ci_vig_ovp[1]], linestyle='-', color='#e87e7e')
axs[1].plot([1, 1], [ovp_ci_vig_spo[0], ovp_ci_vig_spo[1]], linestyle='-', color='#e87e7e')
fig.savefig('position_placement_plt.pdf')
fig.show()

# assessment by treatment groups
fig, vio_plot = plt.subplots()
vio_plot = sns.violinplot(x='partisan_id', y='assessment_vig',
                          hue='treatment_groups', data=voter_data,
                          palette=party_colors)
vio_plot.set_aspect(.25)
vio_plot.set_xlabel('')
vio_plot.set_ylabel('Assessment of asylum law')
plot_range = range(1, 6)
vio_plot.set_yticks(list(plot_range))
vio_plot.set_yticklabels(['very bad' if x == 1 else
                          'very good' if x == 5 else
                          x for x in plot_range])
vio_plot.legend(bbox_to_anchor=(.7, 1), loc=3, borderaxespad=0.)
fig.subplots_adjust(left=.2)
fig.savefig('assessment_plt.pdf')
fig.show()


#########################
# multivariate analyses #
#########################


def params_to_df(res, decimals=2):
    """
    Creates a pandas.core.frame.DataFrame from the parameters of a fitted statsmodels
    regression, along with 95% confidence intervals.
    """
    coeffs = res.params
    ci_lower = res.conf_int()[0]
    ci_upper = res.conf_int()[1]
    results = pd.DataFrame.from_items([('Coefficients', coeffs),
                                       ('Lower CI', ci_lower),
                                       ('Upper CI', ci_upper)])
    return results.round(decimals)


formulas = {'spo': 'spo_pos ~ w2_q23x1 + spo_treatment',
            'ovp': 'ovp_pos ~ w2_q23x2 + spo_treatment',
            'spo_int': 'spo_pos ~ w2_q23x1 + spo_treatment*assessment_vig',
            'ovp_int': 'ovp_pos ~ w2_q23x2 + spo_treatment*assessment_vig'}

ols_results = {}
for model_name, equation in formulas.items():
    mod = ols(equation, voter_data)
    ols_results[model_name] = mod.fit()

for res in ols_results.values():
    print(params_to_df(res).to_latex())


sim = simulate(ols_results['spo'], m=10000)
notreat = sim_predict(sim, setx={'spo_treatment': 0,
                                 'w2_q23x1': voter_data['w2_q23x1'].mean()})
treat = sim_predict(sim, setx={'spo_treatment': 1,
                               'w2_q23x1': voter_data['w2_q23x1'].mean()})
treatment_effect = treat-notreat

sns.violinplot(treatment_effect, orient='v')
plt.show()
