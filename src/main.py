import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
from collections import defaultdict
plt.style.use('seaborn-whitegrid')

from scipy import stats
from sklearn.impute import SimpleImputer

def read_dataset(file):
	return pd.read_csv(file, sep = ';', parse_dates = {'datetime': [0, 2]})

def clean_likes_data(teams):
	dfs = []

	# reads each data file
	for csv in os.listdir('data/likes'):
		df = pd.read_csv('data/likes/' + csv, sep = ';', 
			names = ['city', 'population', 'percentage'])

		df['team'] = teams[int(csv.split('-')[0])]

		dfs.append(df)

	# collapse all data into one dataframe
	likes = pd.concat(dfs)

	# cleans population data
	likes['population'] = likes['population'] \
		.apply(lambda p: p.split(' ')[1].replace('.', '')) \
		.apply(lambda p: '41487' if p == 'NaN' else p) \
		.apply(int)

	# separate city from state
	likes[['city', 'state']] = \
		likes['city'].str.split(', ', n = 1, expand = True)

	# cleans percentage data
	likes['percentage'] = likes['percentage'] \
		.apply(lambda s: s.replace('%', '').replace(',', '.')) \
		.apply(float).apply(lambda n: n / 100.0)

	# estimate amount of fans on each city
	likes['fans'] = (likes['population'] * likes['percentage']).apply(int)

	# remove duplicate rows
	likes = likes[~likes.duplicated(['city', 'state', 'team'])]

	return likes.reset_index(drop = True)

def plot(team, home_matches):
	fig = plt.figure()

	x = np.linspace(1, len(home_matches[team]['attendance']), 
		len(home_matches[team]['attendance']))

	#plt.ylim(0, 50000)

	plt.bar(x, home_matches[team]['attendance'] - a, 
		label = team, tick_label = home_matches[team]['away_team'])
	#plt.bar(x, t2['attendance'] - means.loc['Corinthians']['attendance'], label='Cor')
	#plt.plot(x, data['mean'], label='Médio')
	#plt.plot(x, data['worst'], label='Pior')
	#plt.plot(x, [168] * len(x), linestyle = '--', label='Custo ótimo')

	plt.xlabel('Rodada')
	plt.ylabel('Público')

	#plt.title("Experimento 1")

	plt.legend()

	plt.show()

def main():
	# reads matches data
	data = pd.concat([read_dataset('data/2015.csv'),
		read_dataset('data/2016.csv'),
		read_dataset('data/2017.csv'),
		read_dataset('data/2018.csv')])
	data = data.sort_values('datetime')

	# reads stadium data
	homes = pd.read_csv('data/teams.csv', sep = ';')

	team2state = dict(zip(homes.team, homes.state))
	state2team = defaultdict(list)

	for team, state in team2state.items():
		state2team[state].append(team)

	homes = homes.drop(['stadium'], axis = 1) \
				.join(homes.stadium.str.split(',', expand = True) \
				.stack().reset_index(level = 1, drop = True) \
				.rename('stadium'))
	homes = list(map(lambda r: (r.team, r.stadium), homes.itertuples()))

	# removes matches in uncommon stadiums
	keep = []

	for row in data[['home_team', 'stadium']].itertuples():
		keep.append((row.home_team, row.stadium) in homes)

	data = data[keep]

	# creates auxiliary data structures
	teams = data['home_team'].unique()
	means = {}
	matches = {}
	home_matches = {}
	away_matches = {}
	prices = []
	model1 = []
	model1_count = []

	for team in teams:
		home = data['home_team'] == team
		away = data['away_team'] == team

		matches[team] = data[home | away]
		home_matches[team] = data[home]
		away_matches[team] = data[away]

		m = home_matches[team]

		means[team] = m['attendance'].mean()

		'''prices.append(stats.pearsonr(m['attendance'], 
			m['average_ticket_price'])[0])'''

		model1.append([m[m['away_team'] == t]['attendance'].mean() 
			for t in teams])
		model1_count.append([m[m['away_team'] == t]['attendance'].count() 
			for t in teams])

	likes = clean_likes_data(teams)



	# lists estimated amount and % of supporters by state
	'''likes_by_state = likes.groupby(['team', 'state']).sum()
	likes_by_state['percentage'] = \
		likes_by_state['fans'] / likes_by_state['population']

	#print(likes_by_state.groupby('team').sum())
	#print(likes_by_state.groupby(['state']).sum())
	#print(likes_by_state.loc['Botafogo',])'''



	# list the stadiums each team has played on as home
	#for team, matches in home_matches.items():
	#	print('{};{}'.format(team, ", ".join(home_matches[team]['stadium'].unique())))



	'''model0 = pd.DataFrame.from_dict(means, orient = 'index', columns = ['attendance'])

	model0['attendance'] = model0['attendance'].apply(int)

	print(model0.sort_values('attendance', ascending = False))'''




	#prices = pd.DataFrame(prices, columns = ['p'], 
	#	index = teams).sort_values('p', ascending = False)

	#print(prices)




	model1 = np.matrix(model1)
	model1_count = np.array(model1_count)

	#print(model1)

	imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

	model1 = imp.fit_transform(model1.T).T
	model1 = np.rint(model1)

	#print(model1)



	teams2 = list(teams)
	indexes_to_remove = []

	banned = ['Joinville', 'Goiás', 'América-MG', 'Atlético-GO']

	for team in banned:
		index = teams2.index(team)
		indexes_to_remove.append(index)
		teams2.pop(index)




	home_est = pd.concat([away_matches['América-MG'], 
		away_matches['Atlético-GO']])

	#print(home_est[['home_team', 'attendance']])

	home_est = home_est.groupby(['home_team']).mean().attendance.sort_values(ascending = False).apply(int)

	#print(home_est)

	home_est = list(map(lambda t: int(home_est.loc[t]), teams2))

	#print(home_est)



	away_ests = {}

	'''teams2bkp = list(teams2)

	for team in ['Flamengo', 'Botafogo', 'Vasco', 'Fluminense']:
		teams2 = list(teams2bkp)
		print(team)'''

	for team in teams:
		team_index = int(np.where(teams == team)[0])

		#base = int(means[team])
		base = list(model1[:, team_index].flatten())
		##base_count = list(model1_count[:, team_index].flatten())
		#print(base)
		#print(base_count)
		
		for index in indexes_to_remove:
			base.pop(index)
			##base_count.pop(index)

		away_est = [base[i] - home_est[i] for i in range(len(home_est))]

		#print(away_est)
		#print(team)
		#print(list(zip(teams2, away_est)))

		away_ests[team] = away_est

	model2 = model1.copy()

	for i, home in enumerate(teams):
		for j, away in enumerate(teams):
			if home in banned or away in banned:
				model2[i][j] = np.nan
			else:
				home_index2 = teams2.index(home)
				model2[i][j] = home_est[home_index2] + away_ests[away][home_index2]

	model2 = np.matrix(model2)

	#print(model2)

	'''dists = [0, 2338, 434, 357, 852, 852, 1553, 0, 1043, 357, 1338, 357, 
		1144, 852, 1553, 0, 357, 357, 434, 1553, 2338, 0, 434, 1209,
		1338, 1209, 852, 2805]

	xestados = [0, 357, 434, 852, 1144, 1209, 1338, 1553, 2338, 2805]
	yestados = ['RJ', 'SP', 'MG', 'PR', 'SC', 'BA', 'GO', 'RS', 'PE', 'CE']
	yestados = list(map(lambda s: int(likes_by_state.loc[team, s].fans), yestados))
	print(yestados)
	estados = ['RJ', 'PE', 'MG', 'SP', 'PR', 'PR', 'RS', 'RJ', 'SC', 
		'SP', 'GO', 'SP', 'SC', 'SC', 'PE', 'RJ', 'MG', 'BA', 'GO',
		'BA', 'PR', 'CE']

	away_est.pop(teams2.index(team))
	base_count.pop(teams2.index(team))

	for index in indexes_to_remove:
		dists.pop(index)

	dists.pop(teams2.index(team))
	
	teams2.pop(teams2.index(team))'''





	'''home_ests = {}

	for ho in ['Flamengo', 'Botafogo', 'Vasco', 'Fluminense']:
		home_ests[ho] = []

		for aw in ['Flamengo', 'Botafogo', 'Vasco', 'Fluminense']:
			teams2 = teams2bkp
			print(ho, 'x', aw)

			hoi, awi = int(np.where(teams == ho)[0]), int(np.where(teams == aw)[0])
			
			home_est = int(model1[hoi][awi] - away_est[awi])

			print(home_est)

			home_ests[ho].append(home_est)

	teams2 = list(teams2bkp)'''

	'''''fig, ax1 = plt.subplots()

	#x = np.linspace(0, max(dists) + 200)

	ax1.scatter(dists, away_est, color = 'black')
	#plt.bar(x, t2['attendance'] - means.loc['Corinthians']['attendance'], label='Cor')
	#plt.plot(x, data['mean'], label='Médio')
	#plt.plot(x, data['worst'], label='Pior')
	#plt.plot(x, [168] * len(x), linestyle = '--', label='Custo ótimo')

	for i, txt in enumerate(teams2):
		ax1.annotate('{} ({})'.format(txt, base_count[i]), (dists[i], away_est[i]))

	ax1.set_xlabel('Distância (km)')
	ax1.set_ylabel('Público')

	#plt.title("Experimento 1")

	ax2 = ax1.twinx()

	color = 'r'

	ax2.grid(False)
	ax2.set_ylabel('Estimativa de torcedores', color = color)
	ax2.tick_params(axis = 'y', labelcolor = color)
	ax2.plot(xestados, yestados, color = color)

	fig.tight_layout()

	plt.legend()

	plt.show()'''


	# rmse: model 0 x model 1 x model 2
	m0, m1, m2 = 0, 0, 0
	dev = 0

	for team in teams2:
		filtered_matches = home_matches[team][~home_matches[team]['away_team'].isin(banned)]
		team_index = list(teams).index(team)

		m1_pred = list(map(lambda t: list(teams).index(t), filtered_matches['away_team']))
		m1_pred = list(map(lambda t: model1[team_index, t], m1_pred))
		m1_pred = list(map(int, m1_pred))

		m2_pred = list(map(lambda t: list(teams).index(t), filtered_matches['away_team']))
		m2_pred = list(map(lambda t: model2[team_index, t], m2_pred))
		m2_pred = list(map(int, m2_pred))

		m0 += ((filtered_matches['attendance'] - means[team]) ** 2).sum()
		m1 += ((filtered_matches['attendance'] - m1_pred) ** 2).sum()
		m2 += ((filtered_matches['attendance'] - m2_pred) ** 2).sum()

		dev += ((filtered_matches['attendance'] - filtered_matches['attendance'].mean()) ** 2).sum()

	print('rmse m0', math.sqrt(m0))
	print('rmse m1', math.sqrt(m1))
	print('rmse m2', math.sqrt(m2))

	print('nrmse m0', math.sqrt(m0 / dev))
	print('nrmse m1', math.sqrt(m1 / dev))
	print('nrmse m2', math.sqrt(m2 / dev))

	# rmse: model 0 x model 1 x model 2
	'''media, mediageral = 0, 0

	teams3 = ['Flamengo', 'Botafogo', 'Vasco', 'Fluminense']

	for team in teams3:
		t1 = home_matches[team]['away_team'] == 'Flamengo'
		t2 = home_matches[team]['away_team'] == 'Botafogo'
		t3 = home_matches[team]['away_team'] == 'Vasco'
		t4 = home_matches[team]['away_team'] == 'Fluminense'
		tf = t1 | t2 | t3 | t4

		a = list(map(lambda t: list(teams2).index(t), home_matches[team][tf]['away_team']))
		a = list(map(lambda t: model1[list(teams).index(team)][t], a))
		a = list(map(int, a))

		b1 = list(map(lambda t: list(teams3).index(t), home_matches[team][tf]['away_team']))
		b1 = list(map(lambda t: home_ests[team][t], b1))
		b2 = list(map(lambda t: list(teams3).index(t), home_matches[team][tf]['away_team']))
		b2 = list(map(lambda t: away_ests[t][team], b2))
		b = [b1[i] + b2[i] for i in len(b1)]
		b = list(map(int, b))

		media2 += ((home_matches[team]['attendance'] - b) ** 2).sum()
		media += ((home_matches[team]['attendance'] - a) ** 2).sum()
		mediageral += ((home_matches[team]['attendance'] - means[team]) ** 2).sum()

	print('media', math.sqrt(media))
	print('media geral', math.sqrt(mediageral))'''

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Stopping")
