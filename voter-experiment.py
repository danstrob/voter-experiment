import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.formula.api import ols
from simulation import simulate, sim_predict

# seaborn plot style settings
sns.set(style='darkgrid', color_codes=True, palette='pastel')
party_colors = sns.color_palette(["#5b728a", "#e74c3c"])
export_path = 'exp-latex/'     # path to export images

# path to Stata file
stata_file = 'data/surveydata.dta'

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
voter_data['spo_shift'] = voter_data['spo_pos'] - voter_data[spo_pos_pre]
voter_data['ovp_shift'] = voter_data['ovp_pos'] - voter_data[ovp_pos_pre]
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
partisan_dict = {1: 'SPÖ partisans', 2: 'ÖVP partisans', np.nan: 'Non-partisans/other'}
partisan_dict.update({i: 'Non-partisans/other' for i in range(3, 8)})
voter_data.loc[:, 'partisan_id'] = voter_data['partisan_id'].replace(partisan_dict)
voter_data['spo_partisan'] = pd.get_dummies(voter_data['partisan_id'])['SPÖ partisans']
voter_data['ovp_partisan'] = pd.get_dummies(voter_data['partisan_id'])['ÖVP partisans']

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
axs[0].set_title('SPÖ placement')
axs[1].set_title('ÖVP placement')
axs[0].set_ylabel('Party placement on asylum law')
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
fig.savefig(export_path + 'position_placement_plt.pdf', bbox_inches='tight')
fig.show()

# same thing with barplots
fig, axs = plt.subplots(ncols=2)
sns.factorplot(x="treatment_groups", y="spo_pos", data=voter_data,
               ax=axs[0], kind="bar", palette=party_colors)
sns.factorplot(x="treatment_groups", y="ovp_pos", data=voter_data,
               ax=axs[1], kind="bar", palette=party_colors)
axs[0].set_title('SPÖ placement')
axs[1].set_title('ÖVP placement')
axs[0].set_ylabel('Party placement on asylum law')
axs[1].set_ylabel('')
plot_range = np.arange(0, 11, 2)
means = [spo_mean_vig_ovp, spo_ci_vig_ovp, spo_mean_vig_spo, spo_ci_vig_spo,
         ovp_mean_vig_ovp, ovp_ci_vig_ovp, ovp_mean_vig_spo, ovp_ci_vig_spo]
for ax in axs:
    ax.set_xlabel('')
    ax.set_yticks(list(plot_range))
    for p in ax.patches:
        height = p.get_height()
        m = means.pop(0)
        ll, ul = means.pop(0)
        ax.text(p.get_x() + p.get_width() / 2.,
                height + .5,
                '{:1.2f}\n[{:1.2f},{:1.2f}]'.format(m, ll, ul),
                ha="center", size=10)
axs[0].set_yticklabels(['soften' if x == 0 else
                        'tighten' if x == 10 else
                        x for x in plot_range])
axs[1].set_yticklabels([''])
fig.savefig(export_path + 'position_placement_plt.pdf', bbox_inches='tight')
fig.show()

# same thing with barplots
fig, axs = plt.subplots(ncols=2)
sns.factorplot(x="treatment_groups", y="spo_pos", data=voter_data,
               ax=axs[0], kind="bar", palette=party_colors)
sns.factorplot(x="treatment_groups", y="ovp_pos", data=voter_data,
               ax=axs[1], kind="bar", palette=party_colors)
axs[0].set_title('SPÖ placement')
axs[1].set_title('ÖVP placement')
axs[0].set_ylabel('Party placement on asylum law')
axs[1].set_ylabel('')
plot_range = np.arange(0, 11, 2)
means = [spo_mean_vig_ovp, spo_ci_vig_ovp, spo_mean_vig_spo, spo_ci_vig_spo,
         ovp_mean_vig_ovp, ovp_ci_vig_ovp, ovp_mean_vig_spo, ovp_ci_vig_spo]
for ax in axs:
    ax.set_xlabel('')
    ax.set_yticks(list(plot_range))
    for p in ax.patches:
        height = p.get_height()
        m = means.pop(0)
        ll, ul = means.pop(0)
        ax.text(p.get_x() + p.get_width() / 2.,
                height + .5,
                '{:1.2f}\n[{:1.2f},{:1.2f}]'.format(m, ll, ul),
                ha="center", size=10)
axs[0].set_yticklabels(['soften' if x == 0 else
                        'tighten' if x == 10 else
                        x for x in plot_range])
axs[1].set_yticklabels([''])
fig.savefig(export_path + 'position_placement_plt.pdf', bbox_inches='tight')
fig.show()

# same thing using /shifts/ from previous survey
fig, axs = plt.subplots(ncols=2)
sns.factorplot(x="treatment_groups", y="spo_shift", data=voter_data,
               ax=axs[0], palette=party_colors)
sns.factorplot(x="treatment_groups", y="ovp_shift", data=voter_data,
               ax=axs[1], palette=party_colors)
axs[0].set_title('Shift in SPÖ placement')
axs[1].set_title('Shift in ÖVP placement')
axs[0].set_ylabel('Party placement on asylum law')
axs[1].set_ylabel('')
plot_range = np.arange(-0.6, 0.2, 0.2)
for ax in axs:
    ax.set_xlabel('')
    ax.set_yticks(list(plot_range))
axs[1].yaxis.tick_right()
axs[1].yaxis
fig.savefig(export_path + 'position_shift_plt.pdf', bbox_inches='tight')
fig.show()

# differences in SPÖ and ÖVP placements by treatment and partisan groups
fig, axs = plt.subplots(ncols=2)
sns.factorplot(x='partisan_id', y='spo_pos', kind="bar",
               hue='treatment_groups', data=voter_data, ax=axs[0],
               palette=party_colors)
sns.factorplot(x='partisan_id', y='ovp_pos', kind="bar",
               hue='treatment_groups', data=voter_data, ax=axs[1], legend=False,
               palette=party_colors)
plot_range = np.arange(0, 11, 2)
xlbls = ['Non-\npartisans/\nother', 'ÖVP\npartisans', 'SPÖ\npartisans']
for ax in axs:
    ax.legend('')
    ax.set_xlabel('')
    ax.set_yticks(list(plot_range))
    ax.set_xticklabels(xlbls, {'fontsize': 8})
axs[0].set_yticklabels(['soften' if x == 0 else
                        'tighten' if x == 10 else
                        x for x in plot_range])
axs[0].set_title('SPÖ placement')
axs[1].set_title('ÖVP placement')
axs[0].set_ylabel('Party placement on asylum law')
axs[1].set_ylabel('')
axs[0].legend(title="")
axs[1].legend('')
fig.savefig(export_path + 'partisan_placement_plt.pdf', bbox_inches='tight')
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
fig.savefig(export_path + 'assessment_plt.pdf', bbox_inches='tight')
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
            'spo_int': 'spo_pos ~ w2_q23x1 + spo_treatment*partisan_id',
            'ovp_int': 'ovp_pos ~ w2_q23x2 + spo_treatment*partisan_id'}

ols_results = {}
for model_name, equation in formulas.items():
    mod = ols(equation, voter_data)
    ols_results[model_name] = mod.fit()

# spo predictions (model 1)
spo_treatment0 = {'spo_treatment': 0,
                  'w2_q23x1': voter_data['w2_q23x1'].mean()}
spo_treatment1 = {'spo_treatment': 1,
                  'w2_q23x1': voter_data['w2_q23x1'].mean()}

# ovp predictions (model 2)
ovp_treatment0 = {'spo_treatment': 0,
                  'w2_q23x2': voter_data['w2_q23x2'].mean()}
ovp_treatment1 = {'spo_treatment': 1,
                  'w2_q23x2': voter_data['w2_q23x2'].mean()}

# spo interaction predictions (model 3)
spo_treatment0_non_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 0,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x1': voter_data['w2_q23x1'].mean()}
spo_treatment1_non_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 1,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x1': voter_data['w2_q23x1'].mean()}
spo_treatment0_spo_partisans = {'partisan_id[T.SPÖ partisans]': 1,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 0,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 1,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x1': voter_data['w2_q23x1'].mean()}
spo_treatment1_spo_partisans = {'partisan_id[T.SPÖ partisans]': 1,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 1,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 1,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x1': voter_data['w2_q23x1'].mean()}
spo_treatment0_ovp_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 1,
                                'spo_treatment': 0,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x1': voter_data['w2_q23x1'].mean()}
spo_treatment1_ovp_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 1,
                                'spo_treatment': 1,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 1,
                                'w2_q23x1': voter_data['w2_q23x1'].mean()}

# ovp interaction predictions  (model 4)
ovp_treatment0_non_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 0,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x2': voter_data['w2_q23x2'].mean()}
ovp_treatment1_non_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 1,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x2': voter_data['w2_q23x2'].mean()}
ovp_treatment0_spo_partisans = {'partisan_id[T.SPÖ partisans]': 1,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 0,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 1,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x2': voter_data['w2_q23x2'].mean()}
ovp_treatment1_spo_partisans = {'partisan_id[T.SPÖ partisans]': 1,
                                'partisan_id[T.ÖVP partisans]': 0,
                                'spo_treatment': 1,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 1,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x2': voter_data['w2_q23x2'].mean()}
ovp_treatment0_ovp_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 1,
                                'spo_treatment': 0,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 0,
                                'w2_q23x2': voter_data['w2_q23x2'].mean()}
ovp_treatment1_ovp_partisans = {'partisan_id[T.SPÖ partisans]': 0,
                                'partisan_id[T.ÖVP partisans]': 1,
                                'spo_treatment': 1,
                                'spo_treatment:partisan_id[T.SPÖ partisans]': 0,
                                'spo_treatment:partisan_id[T.ÖVP partisans]': 1,
                                'w2_q23x2': voter_data['w2_q23x2'].mean()}


for res in ols_results.values():
    print(params_to_df(res).to_latex())

# simulate expected values based on model 1 and 2
spo = simulate(ols_results['spo'], m=10000)
ovp = simulate(ols_results['ovp'], m=10000)
spo_t0 = sim_predict(spo, spo_treatment0)
spo_t1 = sim_predict(spo, spo_treatment1)
ovp_t0 = sim_predict(ovp, ovp_treatment0)
ovp_t1 = sim_predict(ovp, ovp_treatment1)

# plot expected values based on model 1 and 2
fig, axs = plt.subplots(ncols=2)
sns.violinplot(data=[spo_t0, spo_t1], ax=axs[0], palette=party_colors)
sns.violinplot(data=[ovp_t0, ovp_t1], ax=axs[1], palette=party_colors)
axs[0].set_title('SPÖ placement (Model 1)')
axs[1].set_title('ÖVP placement (Model 2)')
axs[0].set_ylabel('Party placement on asylum law (expected values)')
xlbls = ['ÖVP\ntreatment', 'SPÖ\ntreatment']
for ax in axs:
    ax.set_xticklabels(xlbls, {'fontsize': 9})
fig.savefig(export_path + 'expected_plt.pdf', bbox_inches='tight')
fig.show()


# spo simulation (model 3)
spo_res = simulate(ols_results['spo_int'], m=10000)
spo_t0_non_partisans = sim_predict(spo_res, spo_treatment0_non_partisans)
spo_t1_non_partisans = sim_predict(spo_res, spo_treatment1_non_partisans)
spo_effect_non_partisans = spo_t1_non_partisans - spo_t0_non_partisans
spo_t0_spo_partisans = sim_predict(spo_res, spo_treatment0_spo_partisans)
spo_t1_spo_partisans = sim_predict(spo_res, spo_treatment1_spo_partisans)
spo_effect_spo_partisans = spo_t1_spo_partisans - spo_t0_spo_partisans
spo_t0_ovp_partisans = sim_predict(spo_res, spo_treatment0_ovp_partisans)
spo_t1_ovp_partisans = sim_predict(spo_res, spo_treatment1_ovp_partisans)
spo_effect_ovp_partisans = spo_t1_ovp_partisans - spo_t0_ovp_partisans

# ovp simulation (model 4)
ovp_res = simulate(ols_results['ovp_int'], m=10000)
ovp_t0_non_partisans = sim_predict(ovp_res, ovp_treatment0_non_partisans)
ovp_t1_non_partisans = sim_predict(ovp_res, ovp_treatment1_non_partisans)
ovp_effect_non_partisans = ovp_t1_non_partisans - ovp_t0_non_partisans
ovp_t0_spo_partisans = sim_predict(ovp_res, ovp_treatment0_spo_partisans)
ovp_t1_spo_partisans = sim_predict(ovp_res, ovp_treatment1_spo_partisans)
ovp_effect_spo_partisans = ovp_t1_spo_partisans - ovp_t0_spo_partisans
ovp_t0_ovp_partisans = sim_predict(ovp_res, ovp_treatment0_ovp_partisans)
ovp_t1_ovp_partisans = sim_predict(ovp_res, ovp_treatment1_ovp_partisans)
ovp_effect_ovp_partisans = ovp_t1_ovp_partisans - ovp_t0_ovp_partisans

# plot first differences by partisan groups based on model 3 and 4
vio_color = sns.color_palette(['#9a9a9a', "#5b728a", "#e74c3c"])
fig, axs = plt.subplots(ncols=2, figsize=(7, 3), sharex=True)
sns.violinplot(data=[spo_effect_non_partisans, spo_effect_ovp_partisans,
                     spo_effect_spo_partisans], ax=axs[0], palette=vio_color, orient='h')
sns.violinplot(data=[ovp_effect_non_partisans, ovp_effect_ovp_partisans,
                     ovp_effect_spo_partisans], ax=axs[1], palette=vio_color, orient='h')
xlbls = ['Non-\npartisans/\nother', 'ÖVP\npartisans', 'SPÖ\npartisans']
plot_range = np.arange(-1, 2, .5)
for ax in axs:
    ax.set_xticks(list(plot_range))
    ax.axvline(zorder=.9)
axs[0].set_title('SPÖ placement (Model 3)')
axs[1].set_title('ÖVP placement (Model 4)')
axs[0].set_yticklabels(xlbls, {'fontsize': 9})
axs[1].set_yticklabels('')
fig.text(0.5, -0.04, 'Treatment effect (first differences)', ha='center')
fig.savefig(export_path + 'effects_plt.pdf', bbox_inches='tight')
fig.show()
