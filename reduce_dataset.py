import os

path = os.getcwd()

f = open(path+'/valid.txt')
lines = f.readlines()

for i in range(0,20):
	line = lines[i].strip()
	open(path+'/mini_data/data/'+line+'-full.png', 'wb').write(open(path+'/data/'+line+'-full.png', 'rb').read())
	open(path+'/mini_data/data/'+line+'.html', 'wb').write(open(path+'/data/'+line+'.html', 'rb').read())


