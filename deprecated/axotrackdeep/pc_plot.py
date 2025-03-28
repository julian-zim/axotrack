import os
from matplotlib import pyplot as plt
import pandas as pd


path = '.\\models'


def plot(dataset):
	features_pc = pd.read_csv(os.path.join(path, f'{dataset}_pc.csv'))
	labels = pd.read_csv(os.path.join(path, f'{dataset}_labels.csv'))

	print(features_pc.head())
	print(labels.head())

	plt.scatter(features_pc['0'], features_pc['1'], c=labels['0'])
	plt.savefig(f'{dataset}.png')
	plt.show()

	

def main():
	plot('D1')
	plot('D2')
	plot('L1')
	plot('roi1')
	plot('roi2')


if __name__ == '__main__':
	main()
