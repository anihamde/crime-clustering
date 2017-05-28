import numpy as np
import csv 
import sys
import random as rando
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

class KMeans(object):
	# K is the K in KMeans
	def __init__(self, K):
		self.K = K

	# X is a (n x j) array where j is the number of columns of interest for each of the n points.
	def fit(self, X):

		n = len(X)
		self.n = n



		centroidmeans = np.array([rando.choice(X)])
		point_nearest_centroid_distance = np.repeat(0.0,self.n)
		#print(centroidmeans)

		while(len(centroidmeans) != self.K):
			for i in range(0,self.n):
				init_diffs_vecs = np.repeat(0.0,len(centroidmeans))
				for k in range(0,len(centroidmeans)):
					diffs_vec = X[i] - centroidmeans[k]
					diffs_vec[0] = min(abs(diffs_vec[0]), 24-abs(diffs_vec[0]))
					# print("SDCZ", diffs_vec)
					# print("SDCZUPINHYO",np.linalg.norm(diffs_vec))
					init_diffs_vecs[k] = np.linalg.norm(diffs_vec)
				# print("HELLO HELLO HELLO", init_diffs_vecs)
				point_nearest_centroid_distance[i] = min(init_diffs_vecs)
				# print(i)

			newarr = np.square(point_nearest_centroid_distance)

			# print("DSF",point_nearest_centroid_distance)

			newarr = newarr/np.sum(newarr)

			# print("QSEZ",newarr)

			# print(np.sum(newarr))
			# print("HERE IT IS",newarr)

			newarr2 = np.repeat(0.0,len(newarr))

			for i in range(0,len(newarr2)):
				newarr2[i] = sum(newarr[0:i])

			newarr2 = [0] + newarr2

			rando_number = rando.uniform(0,1)

			l = 0
			r = len(newarr2)
			while (r - l > 1):
				mid = ((l + r) // 2)

				if (newarr2[mid] > rando_number):
					r = mid

				else:
					l = mid

			#print(X[l])
			centroidmeans = np.vstack((centroidmeans,X[l]))
			#centroidmeans = [X[l]]+centroidmeans

		# print("::::::::::")
		# print(centroidmeans)


		# centroidmeans = X[0:self.K]


		clustermat = np.repeat(0,self.n)
		validator = centroidmeans*3

		countervar = 0

		while(not np.array_equal(validator,centroidmeans)):
			# print(countervar)

			validator = centroidmeans

			for i in range(0,self.n):
				normvals = np.repeat(0.0,self.K)
				for k in range(0,self.K):
					diffs_vec = X[i] - centroidmeans[k]
					diffs_vec[0] = min(abs(diffs_vec[0]), 24-abs(diffs_vec[0]))
					# print("HIFODSHOFI", diffs_vec, X[i], centroidmeans[k])
					normvals[k] = np.linalg.norm(diffs_vec)
				# print("ZZZZZZZZZZZ",normvals)
				clustermat[i] = np.argmin(normvals)
			# print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
			# print(clustermat)
			centroidmeans1 = np.zeros((self.K,3))#X[0:self.K]*0
			count = np.repeat(0,self.K)
			# print(centroidmeans1)
			for i in range(0,self.n):
				centroidmeans1[clustermat[i]] += X[i]
				count[clustermat[i]] += 1

			for s in range(0,self.K):
				centroidmeans1[s] = centroidmeans1[s]/count[s]
				if(np.linalg.norm(centroidmeans1[s]) > 0):
					centroidmeans[s] = centroidmeans1[s]

			countervar += 1

		# print(centroidmeans)

		self.clustermat = clustermat
		self.centroidmeans = centroidmeans
		self.X = X

		self.get_error()

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_centroids(self,matr):
		for i in range(0,self.K):
			print("Mean of class ",i,"!")
			meani = [self.centroidmeans[i][0]*12,self.centroidmeans[i][1]*maxdist,self.centroidmeans[i][2]*maxdist]
			print(meani)
			matr.append(meani)
		return

	def get_error(self):
		sumsq = 0
		for i in range(0,len(self.X)):
			diffs_vec = self.X[i] - self.centroidmeans[self.clustermat[i]]
			diffs_vec[0] = min(abs(diffs_vec[0]), 24-abs(diffs_vec[0]))
			sumsq += (np.linalg.norm(diffs_vec))**2
		print("The average sum of squares error is ",sumsq/len(self.X),"!")
		return


	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	# def create_image_from_array(self, img_array):
	# 	plt.figure()
	# 	plt.imshow(img_array, cmap='Greys_r')
	# 	plt.show()
	# 	return

with open("march2017barenecessities.csv") as f:
	rd = csv.reader(f)
	Z = [r for r in rd]
#np.array([[15,29.79038,-95.4325],[0,29.79273,-95.4077],[18,29.79273,-95.4077],[19,29.80084,-95.4036],[19,29.8007,-95.4142],[15,29.69769,-95.4372],[11,29.63324,-95.4708],[8,29.67679,-95.4339],[7,29.78868,-95.4235],[14,29.64234,-95.3429],[12,29.64216,-95.3661],[13,29.64267,-95.2698],[0,29.61826,-95.4223],[23,29.67869,-95.5662]])
#data in format (lat,long,time)

Z = Z[1:len(Z)]

# print(X[0:5])

for i in range (len(Z)):
	Z[i][0] = float(Z[i][0])
	Z[i][1] = float(Z[i][1])
	Z[i][2] = float(Z[i][2])

# # print(X[0:5])

# Y = Z

# X = []
# for i in range(0,len(Y)):
# 	if Y[i][0] <= 30:
# 		X.append([Y[i][2],Y[i][0],Y[i][1]])

# 	# X[i][0] = Y[i][2]
# 	# X[i][1] = Y[i][0]
# 	# X[i][2] = Y[i][1]

# #Calculating maximum distance

# print('check1')

# # arrayofdist = []

# # for i in range(0,len(X)):
# # 	for j in range(i+1,len(X)):
# # 		arrayofdist = arrayofdist + [((X[i][1]-X[j][1])**2 + (X[i][2]-X[j][2])**2)**(1/2)]

# lat = []
# lon = []

# for i in range(0,len(X)):
# 	lat.append(X[i][1])
# 	lon.append(X[i][2])

# # print(max(lat),min(lat),max(lon),min(lon))

# print('check2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# # max(arrayofdist)
# print(maxdist)

# for i in range(0,len(X)):
# 	X[i][0] = X[i][0]/12
# 	X[i][1] = X[i][1]/maxdist
# 	X[i][2] = X[i][2]/maxdist

# print('check3')
# # print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
# # print(type(int(sys.argv[1])))

# KMeansClassifier = KMeans(K=int(sys.argv[1]))
# print('check4')
# KMeansClassifier.fit(X)
# print('check5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)




# ###### Plotting

# plt.plot(maxdist*np.transpose(X)[1],maxdist*np.transpose(X)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()

# # kmeans = KMeans(n_clusters = , init = "k-means++")

# # kmeans.labels_
# # kmeans.cluster_centers_














with open("mar17.csv") as f:
	rd = csv.reader(f)
	Q = [r for r in rd]

burglary1 = []
autotheft1 = []
theft1 = []
rape1 = []
robbery1 = []
aggravatedassault1 = []
murder1 = []

for i in range(0,len(Z)):
	if(Q[i]==["Burglary"]):
		burglary1.append(Z[i])
	elif(Q[i]==["Auto Theft"]):
		autotheft1.append(Z[i])
	elif(Q[i]==["Theft"]):
		theft1.append(Z[i])
	elif(Q[i]==["Rape"]):
		rape1.append(Z[i])
	elif(Q[i]==["Robbery"]):
		robbery1.append(Z[i])
	elif(Q[i]==["Aggravated Assault"]):
		aggravatedassault1.append(Z[i])
	elif(Q[i]==["Murder"]):
		murder1.append(Z[i])

burglary = []
autotheft = []
theft = []
rape = []
robbery = []
aggravatedassault = []
murder = []

for i in range(0,len(burglary1)):
	if burglary1[i][0] <= 30:
		burglary.append([burglary1[i][2],burglary1[i][0],burglary1[i][1]])
for i in range(0,len(autotheft1)):
	if autotheft1[i][0] <= 30:
		autotheft.append([autotheft1[i][2],autotheft1[i][0],autotheft1[i][1]])
for i in range(0,len(theft1)):
	if theft1[i][0] <= 30:
		theft.append([theft1[i][2],theft1[i][0],theft1[i][1]])
for i in range(0,len(rape1)):
	if rape1[i][0] <= 30:
		rape.append([rape1[i][2],rape1[i][0],rape1[i][1]])
for i in range(0,len(robbery1)):
	if robbery1[i][0] <= 30:
		robbery.append([robbery1[i][2],robbery1[i][0],robbery1[i][1]])
for i in range(0,len(aggravatedassault1)):
	if aggravatedassault1[i][0] <= 30:
		aggravatedassault.append([aggravatedassault1[i][2],aggravatedassault1[i][0],aggravatedassault1[i][1]])
for i in range(0,len(murder1)):
	if murder1[i][0] <= 30:
		murder.append([murder1[i][2],murder1[i][0],murder1[i][1]])








# ###BURGLARY
# print('Burglary 1')

# lat = []
# lon = []
# for i in range(0,len(burglary)):
# 	lat.append(burglary[i][1])
# 	lon.append(burglary[i][2])

# print('Burglary 2')
# print(len(burglary))

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(burglary)):
# 	burglary[i][0] = burglary[i][0]/12
# 	burglary[i][1] = burglary[i][1]/maxdist
# 	burglary[i][2] = burglary[i][2]/maxdist

# print('Burglary 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Burglary 4')
# KMeansClassifier.fit(burglary)

# print('Burglary 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(burglary)[1],maxdist*np.transpose(burglary)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()






# ###Auto Theft
# print('Auto Theft 1')

# lat = []
# lon = []
# for i in range(0,len(autotheft)):
# 	lat.append(autotheft[i][1])
# 	lon.append(autotheft[i][2])

# print('Auto Theft 2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(autotheft)):
# 	autotheft[i][0] = autotheft[i][0]/12
# 	autotheft[i][1] = autotheft[i][1]/maxdist
# 	autotheft[i][2] = autotheft[i][2]/maxdist

# print('Auto Theft 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Auto Theft 4')
# KMeansClassifier.fit(autotheft)

# print('Auto Theft 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(autotheft)[1],maxdist*np.transpose(autotheft)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()



# ###Theft
# print('Auto Theft 1')

# lat = []
# lon = []
# for i in range(0,len(theft)):
# 	lat.append(theft[i][1])
# 	lon.append(theft[i][2])

# print('Theft 2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(theft)):
# 	theft[i][0] = theft[i][0]/12
# 	theft[i][1] = theft[i][1]/maxdist
# 	theft[i][2] = theft[i][2]/maxdist

# print('Theft 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Theft 4')
# KMeansClassifier.fit(theft)

# print('Theft 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(theft)[1],maxdist*np.transpose(theft)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()



# ###Rape
# print('Rape 1')

# lat = []
# lon = []
# for i in range(0,len(rape)):
# 	lat.append(rape[i][1])
# 	lon.append(rape[i][2])

# print('Rape 2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(rape)):
# 	rape[i][0] = rape[i][0]/12
# 	rape[i][1] = rape[i][1]/maxdist
# 	rape[i][2] = rape[i][2]/maxdist

# print('Rape 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Rape 4')
# KMeansClassifier.fit(rape)

# print('Rape 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(rape)[1],maxdist*np.transpose(rape)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()



# ###Robbery
# print('Robbery 1')

# lat = []
# lon = []
# for i in range(0,len(robbery)):
# 	lat.append(robbery[i][1])
# 	lon.append(robbery[i][2])

# print('Robbery 2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(robbery)):
# 	robbery[i][0] = robbery[i][0]/12
# 	robbery[i][1] = robbery[i][1]/maxdist
# 	robbery[i][2] = robbery[i][2]/maxdist

# print('Robbery 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Robbery 4')
# KMeansClassifier.fit(robbery)

# print('Robbery 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(robbery)[1],maxdist*np.transpose(robbery)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()



# ###Aggravated Assault
# print('Aggravated Assault 1')

# lat = []
# lon = []
# for i in range(0,len(aggravatedassault)):
# 	lat.append(aggravatedassault[i][1])
# 	lon.append(aggravatedassault[i][2])

# print('Aggravated Assault 2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(aggravatedassault)):
# 	aggravatedassault[i][0] = aggravatedassault[i][0]/12
# 	aggravatedassault[i][1] = aggravatedassault[i][1]/maxdist
# 	aggravatedassault[i][2] = aggravatedassault[i][2]/maxdist

# print('Aggravated Assault 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Aggravated Assault 4')
# KMeansClassifier.fit(aggravatedassault)

# print('Aggravated Assault 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(aggravatedassault)[1],maxdist*np.transpose(aggravatedassault)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()



# ###Murder
# print('Murder 1')

# lat = []
# lon = []
# for i in range(0,len(murder)):
# 	lat.append(murder[i][1])
# 	lon.append(murder[i][2])

# print('Murder 2')

# maxdist = ((max(lat)-min(lat))**2 + (max(lon)-min(lon))**2)**(1/2)
# print(maxdist)

# for i in range(0,len(murder)):
# 	murder[i][0] = murder[i][0]/12
# 	murder[i][1] = murder[i][1]/maxdist
# 	murder[i][2] = murder[i][2]/maxdist

# print('Murder 3')

# KMeansClassifier = KMeans(K=int(sys.argv[1]))

# print('Murder 4')
# KMeansClassifier.fit(murder)

# print('Murder 5')
# list_of_means = []
# KMeansClassifier.get_centroids(list_of_means)

# ###### Plotting

# plt.plot(maxdist*np.transpose(murder)[1],maxdist*np.transpose(murder)[2],'bo')
# plt.plot(np.transpose(list_of_means)[1],np.transpose(list_of_means)[2],'ro')
# plt.show()









