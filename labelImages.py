import csv

labels = '/Users/wintonyee/Documents/GitHub/ClassifyingEmotionsUsingUnsupervisedLearning/data/Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.csv'
rows = 0

def getYearlyCounts():
	f = open(labels)
	reader = csv.reader(f)
	for row in reader:
		label = []
		FACS = row[2].split('+')
		if '12' in FACS and '23' in FACS:
			label.append('Angry')
		if '9' in FACS or '10' in FACS:
			label.append('Disgust')
		if '1' in FACS and '2' in FACS and ('4' in FACS or '5e' in FACS):
			label.append('Fear')
		if '12' in FACS:
			label.append('Happy')
		if ('1' in FACS and '4' in FACS and '15' in FACS) or ('11' in FACS):
			label.append('Sadness')
		if ('1' in FACS and '2' in FACS) or ('5a' in FACS or '5b' in FACS):
			label.append('Surprise')
		if '14' in FACS:
			label.append('Contempt')
		print(FACS, label)


getYearlyCounts()
