import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from scipy import stats
from sklearn.impute import SimpleImputer

def read_dataset(file):
	return pd.read_csv(file, sep = ';', parse_dates = {'datetime': [0, 2]})

def main():
	data = pd.concat([read_dataset('data/2015.csv'),
		read_dataset('data/2016.csv'),
		read_dataset('data/2017.csv'),
		read_dataset('data/2018.csv')])
	data = data.sort_values('datetime')

	teams = data['home_team'].unique()
	means = {}
	matches = {}
	home_matches = {}
	away_matches = {}
	prices = []
	model1 = []

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

	model0 = pd.DataFrame.from_dict(means, orient = 'index', columns = ['attendance'])

	model0['attendance'] = model0['attendance'].apply(int)

	print(model0.sort_values('attendance', ascending = False))   



	#prices = pd.DataFrame(prices, columns = ['p'], 
	#	index = teams).sort_values('p', ascending = False)

	#print(prices)




	model1 = np.matrix(model1)

	#print(model1)

	imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

	model1 = imp.fit_transform(model1.T).T
	model1 = np.rint(model1)

	#print(model1)




	# rmse: model 0 x model 1
	'''media, mediageral = 0, 0

	for team in teams:
		a = list(map(lambda t: list(teams).index(t), home_matches[team]['away_team']))
		a = list(map(lambda t: model1[list(teams).index(team)][t], a))
		a = list(map(int, a))

		media += ((home_matches[team]['attendance'] - a) ** 2).sum()
		mediageral += ((home_matches[team]['attendance'] - means[team]) ** 2).sum()

	print('media', math.sqrt(media))
	print('media geral', math.sqrt(mediageral))'''





	'''team = 'Botafogo'

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

	plt.show()'''

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Stopping")