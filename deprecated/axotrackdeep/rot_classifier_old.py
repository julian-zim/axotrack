import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from matplotlib import pyplot as plt


# globals
#datapath = 'C:\\Users\\Work\\Desktop\\out\\pca'
datapath = 'C:\\Users\\Zimmermann.admin\\Desktop\\out\\pca'
path = '.\\models'
num_imgs = 4000 # 4000
maxpad = 50


def rescale(value, min_old, max_old, min_new, max_new):
	return (value - min_old) / (max_old - min_old) * (max_new - min_new) + min_new


def make_labels(dataset):
	if dataset == 'roi1':
		ups = [range(5, 581+1), range(584, 584+1), range(1709, 3142+1), range(3542, 3542+1), range(3546, 4004+1)]
		downs = [range(582, 583+1), range(585, 1708+1), range(3143, 3541+1), range(3543, 3545+1)]
	elif dataset == 'roi2':
		ups = [range(7, 795+1), range(3040, 3040+1), range(3048, 3048+1), range(3066, 3066+1), range(3071, 3072+1), range(3074, 4006+1)]
		downs = [range(796, 3039+1), range(3041, 3047+1), range(3049, 3065+1), range(3067, 3070+1), range(3073, 3073+1)]
	else:
		raise ValueError('dataset plz?')

	tags = list()
	for i in range(len(ups) + len(downs)):
		if i % 2 == 0:
			for _ in ups[i // 2]:
				tags.append(0)
		elif True:
			for _ in downs[(i - 1) // 2]:
				tags.append(1)

	labels = pd.DataFrame(data=np.array(tags[:num_imgs]))
	labels.to_csv(os.path.join(path, f'{dataset}_labels.csv'), index=False)


def feature_extraction(dataset):
	data = list()
	start = int(sorted(os.listdir(os.path.join(datapath, dataset)), key=lambda x: int(x.split('.')[0]))[0].split('.')[0])
	for i in range(start, start + num_imgs):
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

		if (i - start) % 100 == 0: print(f'{i - start}/{num_imgs - start}')

	data_frame = pd.DataFrame(data=np.array(data))
	data_frame.to_csv(os.path.join(path, f'{dataset}_features.csv'), index=False)
	print(data_frame.head())


def pc_extraction(dataset):
	# load
	data = pd.read_csv(os.path.join(path, f'{dataset}_features.csv'))

	# transform
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data)

	pca = PCA(n_components=10)
	data_pc = pca.fit_transform(data_scaled)
	print(f'EVR: {pca.explained_variance_ratio_}')

	#plt.plot(pca.explained_variance_ratio_)
	#plt.show()

	# save transformed data
	data_pc_df = pd.DataFrame(data=data_pc)
	data_pc_df.to_csv(os.path.join(path, f'{dataset}_pc.csv'), index=False)

	# save model
	dump(pca, os.path.join(path, f'{dataset}_pca.joblib'))
	dump(scaler, os.path.join(path, f'{dataset}_scaler.joblib'))

	# eigenaxolotl lul
	for i, eigenvector in enumerate(pca.components_):
		max_norm = np.linalg.norm(np.array([255 for _ in eigenvector]))
		max_v = np.max(eigenvector)
		min_v = np.min(eigenvector)
		eigenaxolotl = np.array([rescale(v, min_v, max_v, 0., 255.) for v in eigenvector])
		norm = np.linalg.norm(eigenaxolotl)
		if norm > max_norm: raise ValueError('wtf')

		eigenaxolotl = eigenaxolotl.reshape(maxpad, maxpad)
		cv2.imwrite(os.path.join(path, f'{dataset}_eigen{i}.jpg'), eigenaxolotl)
	

def classification(dataset):
	features_pc = pd.read_csv(os.path.join(path, f'{dataset}_pc.csv'))
	labels = pd.read_csv(os.path.join(path, f'{dataset}_labels.csv'))

	rfc = RandomForestClassifier(n_estimators=1000, random_state=42)
	rfc.fit(features_pc, labels.values.ravel())
	
	dump(rfc, os.path.join(path, f'{dataset}_rfc.joblib'))


def prediction(model, dataset):
	model = load(os.path.join(path, f'{model}_rfc.joblib'))
	features_pc = pd.read_csv(os.path.join(path, f'{dataset}_pc.csv'))
	labels = pd.read_csv(os.path.join(path, f'{dataset}_labels.csv'))
	labels_pred = model.predict(features_pc)
	pd.DataFrame({'labels_pred': labels_pred}).to_csv(os.path.join(path, f'{dataset}_labels_pred.csv'), index=False)
	print(accuracy_score(labels, labels_pred))


if __name__ == '__main__':
	print('')

	#make_labels('roi1')
	#make_labels('roi2')

	#feature_extraction('roi1')
	#feature_extraction('roi2')

	pc_extraction('roi1')
	pc_extraction('roi2')
	
	classification('roi1')
	classification('roi2')

	prediction('roi1', 'roi2')
	prediction('roi2', 'roi1')

	pass
