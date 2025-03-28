import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from matplotlib import pyplot as plt
import utils


# globals
#datapath = 'C:\\Users\\Work\\Desktop\\in\\processed\\InfraRed'
datapath = 'C:\\Users\\Zimmermann.admin\\Desktop\\in\\processed\\InfraRed'
path = '.\\models'
num_imgs = -1 # 4000
maxpad = 50


def join_datasets(dataset1, dataset2):
	features1 = pd.read_csv(os.path.join(path, f'{dataset1}_features.csv'))
	labels1 = pd.read_csv(os.path.join(path, f'{dataset1}_labels.csv'))

	features2 = pd.read_csv(os.path.join(path, f'{dataset2}_features.csv'))
	labels2 = pd.read_csv(os.path.join(path, f'{dataset2}_labels.csv'))

	features12 = pd.concat([features1, features2], axis=0)
	labels12 = pd.concat([labels1, labels2], axis=0)

	labels = pd.DataFrame(data=np.array(labels12))
	labels.to_csv(os.path.join(path, f'{dataset1}-{dataset2}_labels.csv'), index=False)
	features = pd.DataFrame(data=np.array(features12))
	features.to_csv(os.path.join(path, f'{dataset1}-{dataset2}_features.csv'), index=False)


def make_labels(dataset):
	print('MAKING LABELS')
	'''if dataset == 'roi1':
		ups = [range(5, 581+1), range(584, 584+1), range(1709, 3142+1), range(3542, 3542+1), range(3546, 4004+1)]
		downs = [range(582, 583+1), range(585, 1708+1), range(3143, 3541+1), range(3543, 3545+1)]
	elif dataset == 'roi2':
		ups = [range(7, 795+1), range(3040, 3040+1), range(3048, 3048+1), range(3066, 3066+1), range(3071, 3072+1), range(3074, 4006+1)]
		downs = [range(796, 3039+1), range(3041, 3047+1), range(3049, 3065+1), range(3067, 3070+1), range(3073, 3073+1)]'''

	if dataset == 'L1':
		ups = [range(253, 290), range(412, 4954), range(5795, 8064), range(8066, 8067), range(8068, 8069), range(9197, 10631), range(11030, 13401+1)]
		downs = [range(290, 412), range(4954, 5795), range(8064, 8066), range(8067, 8068), range(8069, 9197), range(10631, 11030)]
	elif dataset == 'D1':
		ups = [list(), range(185, 197), range(201, 4239), range(4868, 5378), range(6759, 7032), range(7035, 7036), range(7039, 11850), range(13213, 13511), range(13547, 17872)]
		downs = [range(168, 185), range(197, 201), range(4239, 4868), range(5378, 6759), range(7032, 7035), range(7036, 7039), range(11850, 13213), range(13511, 13547), range(17872, 17995+1)]
	elif dataset == 'D2':
		ups = [list(), range(4053, 4774+1), range(5422, 5529+1), range(5531, 5532+1), range(5584, 5921+1), range(5933, 6858+1), range(6905, 6908+1), range(6911, 6919+1), range(6922, 6924+1), range(6926, 6928+1), range(6932, 6937+1), range(6939, 6941+1), range(6945, 6947+1), range(6950, 6951+1), range(6955, 6956+1), range(6959, 6961+1), range(6963, 6966+1), range(6971, 6978+1), range(6981, 6986+1), range(6991, 6995+1), range(6998, 6999+1), range(7004, 7005+1), range(8821, 12089+1), range(15039, 18008+1)]
		downs = [range(14, 4052+1), range(4775, 5421+1), range(5530, 5530+1), range(5533, 5583+1), range(5922, 5932+1), range(6859, 6904+1), range(6909, 6910+1), range(6920, 6921+1), range(6925, 6925+1), range(6929, 6931+1), range(6938, 6938+1), range(6942, 6944+1), range(6948, 6949+1), range(6952, 6954+1), range(6957, 6958+1), range(6962, 6962+1), range(6967, 6970+1), range(6979, 6980+1), range(6987, 6990+1), range(6996, 6997+1), range(7000, 7003+1), range(7006, 8820+1), range(12090, 15038+1)]
	elif dataset == 'test2':
		ups = [range(124, 162+1), range(164, 167+1)]
		downs = [range(163, 163+1), range(168, 3956+1)]
	elif dataset == 'test4':
		ups = [list(), range(66, 145+1), range(798, 800+1), range(802, 836+1), range(838, 839+1), range(842, 843+1), range(852, 852+1)]
		downs = [range(60, 65+1), range(146, 797+1), range(801, 801+1), range(837, 837+1), range(840, 841+1), range(844, 851+1), range(853, 2383+1)]
	elif dataset == 'test5':
		ups = [range(42, 85+1), range(89, 94+1), range(107, 107+1), range(109, 119+1), range(286, 1409+1)]
		downs = [range(86, 88+1), range(95, 106+1), range(108, 108+1), range(120, 285+1), range(1410, 1717+1)]
	else:
		raise ValueError('dataset plz?')

	tags = list()
	for i in range(len(ups) + len(downs)):
		if i % 2 == 0:
			for _ in ups[i // 2]:
				tags.append(0)
		else:
			for _ in downs[(i - 1) // 2]:
				tags.append(1)

	if num_imgs == -1:
		number = len(tags)
	else:
		number = num_imgs
	labels = pd.DataFrame(data=np.array(tags[:number]))
	labels.to_csv(os.path.join(path, f'{dataset}_labels.csv'), index=False)
	print(labels.head())


def feature_extraction(dataset):
	print('MAKING FEATURES')
	data = list()
	img_names = os.listdir(os.path.join(datapath, dataset))
	if num_imgs == -1:
		number = len(img_names)
	else:
		number = num_imgs
	start = int(sorted(img_names, key=lambda x: int(x.split('.')[0]))[0].split('.')[0])
	for i in range(start, start + number):
		img_name = f'{i}.jpg'
		img = cv2.cvtColor(cv2.imread(os.path.join(datapath, dataset, img_name)), cv2.COLOR_BGR2GRAY)
		#cv2.imshow('opencv', img)
		#cv2.waitKey(1)

		w_pad_val = maxpad - img.shape[1]
		h_pad_val = maxpad - img.shape[0]
		if w_pad_val < 0 or h_pad_val < 0:
			raise ValueError(f'Image {os.path.join(datapath, dataset, img_name)} larger than 50 x 50!')

		img_padded = np.pad(img, ((w_pad_val // 2, w_pad_val - (w_pad_val // 2)), (h_pad_val // 2, h_pad_val - (h_pad_val // 2))))

		img_flat = img_padded.flatten().astype(np.intp)
		data.append(img_flat)

		if (i - start) % 100 == 0: print(f'{i - start}/{number - start}')

	features = pd.DataFrame(data=np.array(data))
	features.to_csv(os.path.join(path, f'{dataset}_features.csv'), index=False)
	print(features.head())


def training(dataset):
	print('TRAINING')
	#features_pc = pd.read_csv(os.path.join(path, f'{dataset}_pc.csv'))
	features = pd.read_csv(os.path.join(path, f'{dataset}_features.csv')).values
	labels = pd.read_csv(os.path.join(path, f'{dataset}_labels.csv')).values.ravel()

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

	mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, verbose=True)
	mlp.fit(X_train, y_train)
	accuracy = mlp.score(X_test, y_test)
	print(accuracy)
	dump(mlp, os.path.join(path, f'{dataset}_mlp.joblib'))


def prediction(model, dataset):
	model = load(os.path.join(path, f'{model}_mlp.joblib'))
	#features_pc = pd.read_csv(os.path.join(path, f'{dataset}_pc.csv'))
	features = pd.read_csv(os.path.join(path, f'{dataset}_features.csv')).values
	labels = pd.read_csv(os.path.join(path, f'{dataset}_labels.csv')).values.ravel()
	labels_pred = model.predict(features)
	pd.DataFrame(data=labels_pred).to_csv(os.path.join(path, f'{dataset}_labels_pred.csv'), index=False)
	print(accuracy_score(labels, labels_pred))


def pc_extraction(dataset, n=10):
	# load
	data = pd.read_csv(os.path.join(path, f'{dataset}_features.csv'))

	# transform
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data)

	pca = PCA(n_components=n)
	data_pc = pca.fit_transform(data_scaled)
	print(f'EVR: {pca.explained_variance_ratio_}')

	plt.plot(pca.explained_variance_ratio_)
	plt.show()

	# save transformed data
	data_pc_df = pd.DataFrame(data=data_pc)
	data_pc_df.to_csv(os.path.join(path, f'{dataset}_pc.csv'), index=False)

	# save model
	#dump(pca, os.path.join(path, f'{dataset}_pca.joblib'))
	#dump(scaler, os.path.join(path, f'{dataset}_scaler.joblib'))

	# eigenaxolotl lul
	for i, eigenvector in enumerate(pca.components_):
		max_norm = np.linalg.norm(np.array([255 for _ in eigenvector]))
		max_v = np.max(eigenvector)
		min_v = np.min(eigenvector)
		eigenaxolotl = np.array([utils.rescale(v, min_v, max_v, 0., 255.) for v in eigenvector])
		norm = np.linalg.norm(eigenaxolotl)
		if norm > max_norm: raise ValueError('wtf')

		eigenaxolotl = eigenaxolotl.reshape(maxpad, maxpad)
		#cv2.imwrite(os.path.join(path, f'{dataset}_eigen{i}.jpg'), eigenaxolotl)


if __name__ == '__main__':
	print('')

	# make_labels('L1')
	# make_labels('D1')
	# make_labels('D2')
	# feature_extraction('L1')
	# feature_extraction('D1')
	# feature_extraction('D2')

	# make_labels('A5')
	# make_labels('A11')
	# make_labels('A12')
	# feature_extraction('A5')
	# feature_extraction('A11')
	# feature_extraction('A12')

	# training('L1')
	# training('D1')
	# training('D2')

	#pc_extraction('L1', n=10)
	#pc_extraction('D1', n=10)
	#pc_extraction('D2', n=10)
	#pc_extraction('roi1', n=10)
	#pc_extraction('roi2', n=10)

	# join_datasets('D1', 'D2')
	# join_datasets('D1', 'L1')
	# join_datasets('D2', 'L1')
	# join_datasets('D1-D2', 'L1')

	# training('D1-D2')
	# training('D1-L1')
	# training('D2-L1')
	# training('D1-D2-L1')

	# prediction('D1-D2', 'L1')
	# prediction('D1-L1', 'D2')
	# prediction('D2-L1', 'D1')

	prediction('test2-test4', 'test5')
	prediction('test2-test5', 'test4')
	prediction('test4-test5', 'test2')

	pass
