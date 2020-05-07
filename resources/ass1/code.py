import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data_1 = scipy.io.loadmat('./Datasets/PET_India.mat')
data_2 = scipy.io.loadmat('./Datasets/rain_India.mat')

data_1 = data_1['PET']
data_2 = data_2['rain']

ori_data_1,ori_data_2 = data_1,data_2

data_1 = np.sum(data_1,axis=2)
data_2 = np.sum(data_2,axis=2)

temp = data_1 - data_2
temp[temp<0] = -1
temp[temp>0] = 1
temp+=1
temp[temp==0]=3

fig = plt.figure(figsize=(25,25))
plt.title("Classification of Water Limited and Energy Limited Zones of India")
im = plt.imshow(temp.T[::-1],cmap=plt.get_cmap('Blues'))
names =['No data','Water Limited','Energy Limited']
values = np.unique(temp.ravel())
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.grid(True)
plt.savefig("./images/Classification of Water Limited and Energy Limited Zones of India.png")

idx = np.where(data_2 == 0)
temp_1,temp_2 = data_1,data_2
temp_1[idx] = 0
temp_2[idx] = 1
temp = np.divide(temp_1,temp_2)

temp[temp>=12]=31
temp[temp<0.375]=31
temp[temp<=0.375] = 30
temp[temp<0.75] = 29
temp[temp<2] = 28
temp[temp<5] = 27
temp[temp<12] = 26
temp-=25
temp[temp==6]=0

fig = plt.figure(figsize=(25,25))
plt.title("Variation of Landscape's Ariadity in India")
im = plt.imshow(temp.T[::-1],cmap=plt.get_cmap('prism'))
names =['No data','Arid','Semi Arid','Sub-Humid','Humid']
values = np.unique(temp.ravel())
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.grid(True)
plt.savefig("./images/Variation of Landscape's Ariadity in India.png")

pet_trends,rain_trends,x_arg = [],[],[]

c = 1951
for i in range(64):
	data_1 = ori_data_1[:,:,np.arange(12)+i*12]
	data_2 = ori_data_2[:,:,np.arange(12)+i*12]
	data_1 = np.sum(data_1,axis=2)
	data_2 = np.sum(data_2,axis=2)

	pet_trends.append(np.sum(data_1))
	rain_trends.append(np.sum(data_2))
	x_arg.append(c)

	temp = data_1 - data_2
	temp[temp<0] = -1
	temp[temp>0] = 1
	temp+=1
	temp[temp==0]=3
	
	fig = plt.figure(figsize=(25,25))
	plt.title(str(c))
	im = plt.imshow(temp.T[::-1],cmap=plt.get_cmap('Blues'))
	names =['No data','Water Limited','Energy Limited']
	values = np.unique(temp.ravel())
	colors = [ im.cmap(im.norm(value)) for value in values]
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
	plt.grid(True)
	plt.savefig("./images/Temporal Variability/P-PET/"+str(c)+".png")
	
	idx = np.where(data_2 == 0)
	temp_1,temp_2 = data_1,data_2
	temp_1[idx] = 0
	temp_2[idx] = 1
	temp = np.divide(temp_1,temp_2)
	
	temp[temp>=12]=31
	temp[temp<0.375]=31
	temp[temp<=0.375] = 30
	temp[temp<0.75] = 29
	temp[temp<2] = 28
	temp[temp<5] = 27
	temp[temp<12] = 26
	temp-=25
	temp[temp==6]=0

	fig = plt.figure(figsize=(25,25))
	plt.title(str(c))
	im = plt.imshow(temp.T[::-1],cmap=plt.get_cmap('prism'))
	names =['No data','Arid','Semi Arid','Sub-Humid','Humid']
	values = np.unique(temp.ravel())
	colors = [ im.cmap(im.norm(value)) for value in values]
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
	plt.grid(True)
	plt.savefig("./images/Temporal Variability/PET_by_P/"+str(c)+".png")
	c += 1

fig = plt.figure(figsize=(40,20))
plt.title('Rain Trends in India')
plt.plot(x_arg,rain_trends)
plt.savefig("./images/Rain Trends in India.png")

fig = plt.figure(figsize=(40,20))
plt.title('PET Trends in India')
plt.plot(x_arg,pet_trends)
plt.savefig("./images/PET Trends in India.png")

fig = plt.figure(figsize=(40,20))
plt.title('P-PET Trends in India')
plt.plot(x_arg,np.asarray(rain_trends) - np.asarray(pet_trends))
plt.savefig("./images/P-PET Trends in India.png")

fig = plt.figure(figsize=(40,20))
plt.title('PET/P Trends in India')
plt.plot(x_arg,np.divide(pet_trends,rain_trends))
plt.savefig("./images/PET_by_P Trends in India.png")

c = 1951
for i in range(7):
	if i!= 6:
		data_1 = ori_data_1[:,:,np.arange(120)+i*120]
		data_2 = ori_data_2[:,:,np.arange(120)+i*120]
	else:
		data_1 = ori_data_1[:,:,np.arange(48)+i*120]
		data_2 = ori_data_2[:,:,np.arange(48)+i*120]

	data_1 = np.sum(data_1,axis=2)
	data_2 = np.sum(data_2,axis=2)

	temp = data_1 - data_2
	temp[temp<0] = -1
	temp[temp>0] = 1
	temp+=1
	temp[temp==0]=3
	
	fig = plt.figure(figsize=(25,25))
	plt.title(str(c)+"-"+str(min(c+9,2014)))
	im = plt.imshow(temp.T[::-1],cmap=plt.get_cmap('Blues'))
	names =['No data','Water Limited','Energy Limited']
	values = np.unique(temp.ravel())
	colors = [ im.cmap(im.norm(value)) for value in values]
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
	plt.grid(True)
	plt.savefig("./images/Spatial Variability/P-PET/"+str(c)+"-"+str(min(c+9,2014))+".png")

	if i==0 :
		prev_1 = temp
	else:
		diff = temp - prev_1
		fig = plt.figure()
		plt.imshow(diff.T[::-1])
		plt.title("Decade Difference "+str(i))
		plt.grid(True)
		plt.savefig("./images/Spatial Variability/P-PET/"+"Decade Difference "+str(i)+".png")


	idx = np.where(data_2 == 0)
	temp_1,temp_2 = data_1,data_2
	temp_1[idx] = 0
	temp_2[idx] = 1
	temp = np.divide(temp_1,temp_2)
	
	temp[temp>=12]=31
	temp[temp<0.375]=31
	temp[temp<=0.375] = 30
	temp[temp<0.75] = 29
	temp[temp<2] = 28
	temp[temp<5] = 27
	temp[temp<12] = 26
	temp-=25
	temp[temp==6]=0

	fig = plt.figure(figsize=(25,25))
	plt.title(str(c)+"-"+str(min(c+9,2014)))
	im = plt.imshow(temp.T[::-1],cmap=plt.get_cmap('prism'))
	names =['No data','Arid','Semi Arid','Sub-Humid','Humid']
	values = np.unique(temp.ravel())
	colors = [ im.cmap(im.norm(value)) for value in values]
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
	plt.grid(True)
	plt.savefig("./images/Spatial Variability/PET_by_P/"+str(c)+"-"+str(min(c+9,2014))+".png")

	if i==0 :
		prev_2 = temp
	else:
		diff = temp - prev_2
		fig = plt.figure()
		plt.imshow(diff.T[::-1])
		plt.title("Decade Difference "+str(i))
		plt.grid(True)
		plt.savefig("./images/Spatial Variability/PET_by_P/"+"Decade Difference "+str(i)+".png")

	c += 10