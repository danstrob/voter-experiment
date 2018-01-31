# stacked barplot for assessment of vignette
crosstab_spo = pd.crosstab(
    voter_data[assessment_vig_spo], voter_data['partisan_id'], margins=True)
crosstab_ovp = pd.crosstab(
    voter_data[assessment_vig_ovp], voter_data['partisan_id'], margins=True)

y = {}
for i in range(5):
    ovpv = crosstab_ovp.iloc[i, :3] / crosstab_ovp.iloc[5, :3]
    spov = crosstab_spo.iloc[i, :3] / crosstab_spo.iloc[5, :3]
    y[i] = sum(map(list, zip(ovpv, spov)), [])

ind = [0, 1, 3, 4, 6, 7]
fig, ax = plt.subplots()
bar1 = ax.bar(ind, y[0], color='#083c7d')
bar2 = ax.bar(ind, y[1], bottom=y[0], color='#3b8bc2')
bottom = [i + j for i, j in zip(y[0], y[1])]
bar3 = ax.bar(ind, y[2], bottom=bottom, color='#c6dbef')
bottom = [i + j for i, j in zip(bottom, y[2])]
bar4 = ax.bar(ind, y[3], bottom=bottom, color='#fc8767')
bottom = [i + j for i, j in zip(bottom, y[3])]
bar5 = ax.bar(ind, y[4], bottom=bottom, color='#7c0510')
fig.legend([bar1, bar2, bar3, bar4, bar5],
           ['very good', 'good', 'so-so', 'bad', 'very bad'],
           bbox_to_anchor=(.8, 1), ncol=3, fancybox=True, shadow=True)
ax.set_ylabel('Percentage')
plt.xticks([.5, 3.5, 6.5], ['Non-partisan/other',
                            'SPÖ partisan', 'ÖVP partisan'])
plt.savefig(export_path + 'rating_barchart.pdf', bbox_inches='tight')
plt.show()





ass_n = np.arange(1, 6)
treat_n = np.repeat([1], 5)
pre = np.full(5, voter_data[spo_pos_pre].mean())
Xnew = np.column_stack((ass_n, treat_n, pre))
ynewpred = res.predict(Xnew)  # predict out of sample
print(ynewpred)
