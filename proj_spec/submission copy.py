import helper
import pandas as pd
import numpy as np
import pickle
from sklearn import tree, model_selection, naive_bayes
from sklearn.metrics import f1_score

################# training #################

def train(data, classifier_file):# do not change the heading of the function , classifier_file
	vowel = ['AH', 'ER', 'IY', 'IH', 'OW', 'AY', 'AA', 'AW', 'UH', 'UW', 'EY', 'OY', 'AO', 'EH', 'AE']
	vowelnumber = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	consonant = ['P','B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','R','S','SH','T','TH','V','W','Y','Z','ZH']
	stresses = ['0', '1', '2']

	attributes = ['last',  'vowelnum', 'combination', 'hou',  'position_stress']#'position_stress'
	features = ['last',  'vowelnum',  'combination', 'hou']
	instances = []

	houzhui = {'LLS': 48, 'RAS': 40, 'IKE': 44, 'DES': 79, 'ISM': 76, 'ARS': 45, 'PER': 110, 'TRA': 33, 'SLY': 50, 'MON': 71, 'INA': 122, 'ERY': 98, 'OUT': 35, 'HAM': 95, 'GED': 52, 'IFY': 43, 'TER': 498, 'NGS': 148, 'BLE': 278, 'NTS': 199, 'NDA': 69, 'NIC': 57, 'ORD': 141, 'URE': 81, 'LOW': 72, 'ALD': 30, 'LLE': 148, 'VEY': 37, 'ELD': 116, 'LES': 247, 'ERT': 191, 'INI': 76, 'REY': 44, 'ATO': 40, 'EER': 32, 'DON': 63, 'DED': 149, 'SKI': 333, 'AMA': 40, 'RLY': 62, 'FER': 86, 'IVE': 150, 'GHT': 70, 'CAL': 61, 'IER': 225, 'ILL': 104, 'MEN': 66, 'ITZ': 55, 'IAL': 75, 'ANS': 161, 'ATS': 31, 'ATE': 340, 'IZE': 111, 'ONA': 32, 'NER': 452, 'DIE': 43, 'LED': 179, 'NNA': 41, 'RAN': 57, 'ICE': 65, 'RER': 80, 'ARE': 59, 'NAL': 55, 'KEY': 60, 'ASE': 38, 'ERO': 54, 'RIA': 45, 'ICS': 95, 'ILY': 35, 'ARO': 55, 'RTS': 35, 'OUS': 205, 'ARY': 85, 'ULT': 31, 'HEN': 40, 'BER': 115, 'EAD': 51, 'ONG': 38, 'AND': 236, 'ENA': 57, 'LIS': 63, 'LAN': 57, 'BEE': 35, 'CES': 101, 'BEL': 62, 'ANA': 84, 'STA': 35, 'RIE': 49, 'SED': 95, 'ECT': 30, 'ZED': 115, 'ION': 514, 'ENE': 59, 'TED': 466, 'KLE': 39, 'OCK': 103, 'ADO': 32, 'ARA': 52, 'SER': 168, 'KIN': 117, 'CTS': 50, 'ZES': 49, 'DEN': 124, 'HIP': 43, 'ORA': 44, 'SEN': 130, 'HER': 235, 'KED': 37, 'ULA': 33, 'CAN': 31, 'ESS': 351, 'EAU': 71, 'NIA': 55, 'ETH': 31, 'OSE': 34, 'HES': 115, 'ITO': 39, 'COM': 30, 'GLE': 65, 'TEN': 86, 'NTE': 43, 'ALA': 39, 'RIS': 37, 'AIR': 34, 'ESE': 47, 'SCO': 51, 'MAR': 39, 'IUM': 54, 'ISH': 131, 'ALS': 90, 'VER': 116, 'OME': 41, 'UGH': 89, 'OWN': 52, 'ANO': 135, 'ANK': 34, 'IRE': 75, 'GEL': 42, 'VEN': 41, 'EEN': 44, 'ANE': 60, 'USE': 60, 'ELL': 330, 'LON': 54, 'IST': 166, 'RES': 129, 'SEY': 60, 'TTE': 151, 'RTH': 68, 'TLY': 102, 'NDS': 61, 'WER': 51, 'SES': 179, 'WIN': 34, 'ORS': 133, 'EAN': 30, 'ISE': 60, 'ITA': 48, 'ORY': 43, 'ORE': 111, 'TON': 404, 'NET': 37, 'ITY': 113, 'IED': 81, 'VED': 32, 'NED': 139, 'ELA': 37, 'CKI': 50, 'ADA': 30, 'NAN': 52, 'NIS': 35, 'GES': 98, 'KEL': 55, 'ILE': 48, 'RON': 100, 'DLY': 44, 'CCI': 42, 'ONS': 362, 'CED': 39, 'TAN': 37, 'ANI': 62, 'LLO': 117, 'ARD': 255, 'ITS': 51, 'HAN': 94, 'ERE': 44, 'ERI': 36, 'EES': 36, 'AYS': 47, 'LIO': 32, 'EIN': 88, 'NCE': 174, 'OUR': 30, 'OLD': 72, 'ZER': 105, 'DIN': 41, 'ITE': 86, 'NON': 44, 'RAL': 57, 'GLY': 62, 'INS': 137, 'ERS': 911, 'NAS': 31, 'ADE': 35, 'OTT': 45, 'IES': 323, 'RTY': 39, 'ERN': 38, 'ORN': 45, 'HEY': 31, 'DGE': 63, 'ATH': 30, 'SSA': 30, 'ROS': 37, 'BLY': 49, 'TIN': 54, 'ANT': 142, 'KER': 269, 'PLE': 34, 'MAN': 784, 'VES': 51, 'MEL': 42, 'ACK': 91, 'KEN': 73, 'IUS': 39, 'TEL': 62, 'RIO': 34, 'ECK': 50, 'LIA': 63, 'ONT': 34, 'ICA': 47, 'ZEL': 44, 'ETS': 84, 'RAM': 41, 'VIN': 41, 'INE': 257, 'DAY': 33, 'GAR': 30, 'SEL': 56, 'AGE': 110, 'NCY': 50, 'UND': 55, 'RNE': 33, 'ENS': 109, 'EDA': 33, 'NEY': 177, 'TLE': 68, 'NNE': 35, 'REE': 33, 'TIC': 132, 'ONI': 36, 'IDE': 69, 'ERA': 49, 'LAR': 46, 'ELY': 118, 'GER': 437, 'RDS': 64, 'TRY': 41, 'UER': 50, 'ERG': 162, 'IEL': 31, 'ITT': 46, 'OLA': 77, 'INO': 133, 'WAY': 78, 'NIE': 41, 'TES': 249, 'LIN': 166, 'ARK': 38, 'TOR': 105, 'ALE': 77, 'ENT': 294, 'DEL': 56, 'OFF': 106, 'ART': 97, 'RED': 208, 'ING': 2430, 'ROW': 48, 'DLE': 54, 'LET': 55, 'OWS': 32, 'LER': 504, 'ROM': 35, 'REN': 72, 'END': 31, 'TAS': 35, 'THY': 32, 'ICH': 139, 'ACH': 71, 'TTA': 48, 'NOS': 30, 'MER': 185, 'ELS': 70, 'URG': 64, 'LEY': 396, 'LEN': 75, 'LIE': 46, 'ETT': 147, 'YER': 101, 'TIS': 45, 'GAN': 83, 'GEN': 65, 'AIN': 68, 'ICK': 218, 'URN': 44, 'CKS': 50, 'LLI': 92, 'URY': 41, 'RIN': 66, 'OLE': 30, 'HED': 51, 'SKY': 95, 'TAL': 58, 'EST': 183, 'FUL': 63, 'TTI': 59, 'STS': 126, 'NTO': 31, 'ONE': 188, 'OOD': 114, 'LLA': 180, 'IAN': 189, 'ANN': 120, 'ACE': 45, 'OOK': 50, 'PEL': 31, 'ALL': 115, 'DER': 305, 'LLY': 179, 'RIC': 44, 'UTS': 32, 'MED': 47, 'OND': 38, 'SON': 497, 'NES': 110, 'RRY': 81, 'CHI': 36, 'NEN': 30}
##        
	english = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
			   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

	
                
	for i in range(len(data)):
		vowelnum = 0
		consonantnum = 0
		together = data[i]
		index = together.index(':')
		sound = together[index+1:]   # get the pronunciation part
		phoneme = sound.split(' ')
		word = together[:index]    # get the corresponding word
		combination = '9'


		if word[-1] == 'S':
			word = word[:-1]
		elif word[-1:] == 'D':
			word = word[:-1]

		if len(word) < 3:
			hou = 0
		else:
			if word[-3:] in houzhui:
				hou = ((english.index(word[-3]) + 1) * 100 + (english.index(word[-2]) + 1)) * 100 + english.index(word[-1])+1
			else:
				hou = 0

                
		# to get the features based on the number of vowel
		number_vowel = 0
		contain2 = []
		word_suffix = 0
		for i in range(len(phoneme)):
			if phoneme[i][-1] in stresses:
				contain2.append(phoneme[i][:-1])
				number_vowel = number_vowel + 1
		if len(word) < 3:
			if len(word) == 1:
				word_suffix = english.index(word[0]) + 1
			elif len(word) == 2:
				word_suffix = (english.index(word[0]) + 1) * 100 + english.index(word[1]) + 1
		else:
#			word_suffix = ((english.index(word[-3]) + 1) * 100 + (english.index(word[-2]) + 1)) * 100 + english.index(word[-1])+1
			word_suffix = (english.index(word[-2]) + 1) * 100 + english.index(word[-1]) + 1

#		if number_vowel == 1:
#			word_suffix = '1' + str(english.index(newword[-2]) + 50) + str(english.index(newword[-1]) + 50)
#		elif number_vowel == 2:
#			word_suffix = '2' + str(english.index(newword[-2]) + 50) + str(english.index(newword[-1]) + 50)
#		elif number_vowel == 3:
#			word_suffix = '3' + str(english.index(newword[-2]) + 50) + str(english.index(newword[-1]) + 50)
#		else:
#			word_suffix = '4' + str(english.index(newword[-2]) + 50) + str(english.index(newword[-1]) + 50)


		# get the last feature
		if number_vowel == 1:
			last = vowel.index(contain2[0]) + 1
		else:
			last = (vowel.index(contain2[-2])+1) * 100
			last = last + vowel.index(contain2[-1])+1
			
#		if number_vowel == 1:
#			last = '1' + str(vowel.index(contain2[-1]) + 50)
#		elif number_vowel == 2:
#			last = '2' + str(vowel.index(contain2[-2]) + 50) + str(vowel.index(contain2[-1]) + 50)
#		elif number_vowel == 3:
#			last = '3' + str(vowel.index(contain2[-2]) + 50) + str(vowel.index(contain2[-1]) + 50)
#		else:
#			last = '4' + str(vowel.index(contain2[-2]) + 50) + str(vowel.index(contain2[-1]) + 50)

		position_stress = 1
		contain = []
		
		for j in range(len(phoneme)):
			if phoneme[j][-1] in stresses:
				contain.append(phoneme[j][:-1])
				vowelnum = vowelnum + 1
				if phoneme[j][-1] == '1':
					pos = vowel.index(phoneme[j][:-1])
					primary_vowel = vowelnumber[pos]
					position_stress = vowelnum
					if j == 0:
						prefix = 0
					else:
						if phoneme[j - 1][-1] in stresses:
							pos2 = vowel.index(phoneme[j - 1][:-1])
							prefix = 100 + pos2
						else:
							prefix = consonant.index(phoneme[j - 1]) + 1
				location = vowel.index(phoneme[j][:-1]) + 1
				if location < 10:
					combination = combination[0] + '0' + str(location) + combination[1:]
				else:
					combination = combination[0] + str(location) + combination[1:]

			else:
				consonantnum = consonantnum + 1
##		print(combination)

                
		instances.append([last, vowelnum,  combination,  hou, position_stress])#position_stress
	df = pd.DataFrame(data=instances, columns=attributes)
	x = df[features]
	y = df.position_stress
	
#	y = df.inverse_position
#	print(df.head())
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)#, random_state=0

	clf = tree.DecisionTreeClassifier(criterion='entropy')
	model = clf.fit(x_train, y_train)
##	print(model.feature_importances_)
#	clf = naive_bayes.MultinomialNB(alpha=0.01)
#	model = clf.fit(x_train, y_train)
#	clf = KNeighborsClassifier()
#	model = clf.fit(x_train, y_train)
##	print(model.feature_importances_)

	# evaluations
#	print(model.score(x_train, y_train))
#	print(model.score(x_test,y_test))
#	print(dict)
#	print(probability)

	prediction = list(clf.predict(x_test))
	ground_truth = list(y_test)



	precision = [0,0,0,0]
	count = [0,0,0,0]
	for i in range(len(ground_truth)):
		if ground_truth[i] == prediction[i]:
			precision[ground_truth[i] - 1] +=1
		count[ground_truth[i] - 1] += 1
	for j in range(len(count)):
		precision[j] = precision[j]/count[j]
	print(count)
	print(precision)


#	scores = cross_val_score(clf, x, y, cv=5,scoring='f1_macro')
#	print(scores)


	print(f1_score(ground_truth, prediction, average='macro'))
#	return f1_score(ground_truth, prediction, average='macro')
#	print(df.head())


##	output = open(classifier_file, 'wb')
##	pickle.dump(clf, output)
##	output.close()
##
##
################### testing #################
##
##def test(data, classifier_file):# do not change the heading of the function
##	file = open(classifier_file, 'rb')
##	clf = pickle.load(file)
##
##	vowel = ['AH', 'ER', 'IY', 'IH', 'OW', 'AY', 'AA', 'AW', 'UH', 'UW', 'EY', 'OY', 'AO', 'EH', 'AE']
##	vowelnumber = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
##	consonant = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH',
##				 'V', 'W', 'Y', 'Z', 'ZH']
##	features = ['last', 'vowelnum', 'combination','hou']
##	english = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
##			   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
##	houzhui = {'LLS': 48, 'RAS': 40, 'IKE': 44, 'DES': 79, 'ISM': 76, 'ARS': 45, 'PER': 110, 'TRA': 33, 'SLY': 50, 'MON': 71, 'INA': 122, 'ERY': 98, 'OUT': 35, 'HAM': 95, 'GED': 52, 'IFY': 43, 'TER': 498, 'NGS': 148, 'BLE': 278, 'NTS': 199, 'NDA': 69, 'NIC': 57, 'ORD': 141, 'URE': 81, 'LOW': 72, 'ALD': 30, 'LLE': 148, 'VEY': 37, 'ELD': 116, 'LES': 247, 'ERT': 191, 'INI': 76, 'REY': 44, 'ATO': 40, 'EER': 32, 'DON': 63, 'DED': 149, 'SKI': 333, 'AMA': 40, 'RLY': 62, 'FER': 86, 'IVE': 150, 'GHT': 70, 'CAL': 61, 'IER': 225, 'ILL': 104, 'MEN': 66, 'ITZ': 55, 'IAL': 75, 'ANS': 161, 'ATS': 31, 'ATE': 340, 'IZE': 111, 'ONA': 32, 'NER': 452, 'DIE': 43, 'LED': 179, 'NNA': 41, 'RAN': 57, 'ICE': 65, 'RER': 80, 'ARE': 59, 'NAL': 55, 'KEY': 60, 'ASE': 38, 'ERO': 54, 'RIA': 45, 'ICS': 95, 'ILY': 35, 'ARO': 55, 'RTS': 35, 'OUS': 205, 'ARY': 85, 'ULT': 31, 'HEN': 40, 'BER': 115, 'EAD': 51, 'ONG': 38, 'AND': 236, 'ENA': 57, 'LIS': 63, 'LAN': 57, 'BEE': 35, 'CES': 101, 'BEL': 62, 'ANA': 84, 'STA': 35, 'RIE': 49, 'SED': 95, 'ECT': 30, 'ZED': 115, 'ION': 514, 'ENE': 59, 'TED': 466, 'KLE': 39, 'OCK': 103, 'ADO': 32, 'ARA': 52, 'SER': 168, 'KIN': 117, 'CTS': 50, 'ZES': 49, 'DEN': 124, 'HIP': 43, 'ORA': 44, 'SEN': 130, 'HER': 235, 'KED': 37, 'ULA': 33, 'CAN': 31, 'ESS': 351, 'EAU': 71, 'NIA': 55, 'ETH': 31, 'OSE': 34, 'HES': 115, 'ITO': 39, 'COM': 30, 'GLE': 65, 'TEN': 86, 'NTE': 43, 'ALA': 39, 'RIS': 37, 'AIR': 34, 'ESE': 47, 'SCO': 51, 'MAR': 39, 'IUM': 54, 'ISH': 131, 'ALS': 90, 'VER': 116, 'OME': 41, 'UGH': 89, 'OWN': 52, 'ANO': 135, 'ANK': 34, 'IRE': 75, 'GEL': 42, 'VEN': 41, 'EEN': 44, 'ANE': 60, 'USE': 60, 'ELL': 330, 'LON': 54, 'IST': 166, 'RES': 129, 'SEY': 60, 'TTE': 151, 'RTH': 68, 'TLY': 102, 'NDS': 61, 'WER': 51, 'SES': 179, 'WIN': 34, 'ORS': 133, 'EAN': 30, 'ISE': 60, 'ITA': 48, 'ORY': 43, 'ORE': 111, 'TON': 404, 'NET': 37, 'ITY': 113, 'IED': 81, 'VED': 32, 'NED': 139, 'ELA': 37, 'CKI': 50, 'ADA': 30, 'NAN': 52, 'NIS': 35, 'GES': 98, 'KEL': 55, 'ILE': 48, 'RON': 100, 'DLY': 44, 'CCI': 42, 'ONS': 362, 'CED': 39, 'TAN': 37, 'ANI': 62, 'LLO': 117, 'ARD': 255, 'ITS': 51, 'HAN': 94, 'ERE': 44, 'ERI': 36, 'EES': 36, 'AYS': 47, 'LIO': 32, 'EIN': 88, 'NCE': 174, 'OUR': 30, 'OLD': 72, 'ZER': 105, 'DIN': 41, 'ITE': 86, 'NON': 44, 'RAL': 57, 'GLY': 62, 'INS': 137, 'ERS': 911, 'NAS': 31, 'ADE': 35, 'OTT': 45, 'IES': 323, 'RTY': 39, 'ERN': 38, 'ORN': 45, 'HEY': 31, 'DGE': 63, 'ATH': 30, 'SSA': 30, 'ROS': 37, 'BLY': 49, 'TIN': 54, 'ANT': 142, 'KER': 269, 'PLE': 34, 'MAN': 784, 'VES': 51, 'MEL': 42, 'ACK': 91, 'KEN': 73, 'IUS': 39, 'TEL': 62, 'RIO': 34, 'ECK': 50, 'LIA': 63, 'ONT': 34, 'ICA': 47, 'ZEL': 44, 'ETS': 84, 'RAM': 41, 'VIN': 41, 'INE': 257, 'DAY': 33, 'GAR': 30, 'SEL': 56, 'AGE': 110, 'NCY': 50, 'UND': 55, 'RNE': 33, 'ENS': 109, 'EDA': 33, 'NEY': 177, 'TLE': 68, 'NNE': 35, 'REE': 33, 'TIC': 132, 'ONI': 36, 'IDE': 69, 'ERA': 49, 'LAR': 46, 'ELY': 118, 'GER': 437, 'RDS': 64, 'TRY': 41, 'UER': 50, 'ERG': 162, 'IEL': 31, 'ITT': 46, 'OLA': 77, 'INO': 133, 'WAY': 78, 'NIE': 41, 'TES': 249, 'LIN': 166, 'ARK': 38, 'TOR': 105, 'ALE': 77, 'ENT': 294, 'DEL': 56, 'OFF': 106, 'ART': 97, 'RED': 208, 'ING': 2430, 'ROW': 48, 'DLE': 54, 'LET': 55, 'OWS': 32, 'LER': 504, 'ROM': 35, 'REN': 72, 'END': 31, 'TAS': 35, 'THY': 32, 'ICH': 139, 'ACH': 71, 'TTA': 48, 'NOS': 30, 'MER': 185, 'ELS': 70, 'URG': 64, 'LEY': 396, 'LEN': 75, 'LIE': 46, 'ETT': 147, 'YER': 101, 'TIS': 45, 'GAN': 83, 'GEN': 65, 'AIN': 68, 'ICK': 218, 'URN': 44, 'CKS': 50, 'LLI': 92, 'URY': 41, 'RIN': 66, 'OLE': 30, 'HED': 51, 'SKY': 95, 'TAL': 58, 'EST': 183, 'FUL': 63, 'TTI': 59, 'STS': 126, 'NTO': 31, 'ONE': 188, 'OOD': 114, 'LLA': 180, 'IAN': 189, 'ANN': 120, 'ACE': 45, 'OOK': 50, 'PEL': 31, 'ALL': 115, 'DER': 305, 'LLY': 179, 'RIC': 44, 'UTS': 32, 'MED': 47, 'OND': 38, 'SON': 497, 'NES': 110, 'RRY': 81, 'CHI': 36, 'NEN': 30}
##	instance = []
##
##	for i in range(len(data)):
##		vowelnum = 0
##		consonantnum = 0
##		together = data[i]
##		index = together.index(':')
##		sound = together[index + 1:]  # get the pronunciation part
##		phoneme = sound.split(' ')
##		word2 = together[:index]  # get the corresponding word
##		combination = '9'
##
##		if word2[-1] == 'S':
##			word2 = word2[:-1]
##		elif word2[-1:] == 'D':
##			word2 = word2[:-1]
##		# to get the features based on the number of vowel
##
##		if len(word2) < 3:
##			hou = 0
##		else:
##			if word2[-3:] in houzhui:
##				hou = ((english.index(word2[-3]) + 1) * 100 + (english.index(word2[-2]) + 1)) * 100 + english.index(word2[-1])+1
##			else:
##				hou = 0
##
##		number_vowel = 0
##		contain2 = []
##
##		for i in range(len(phoneme)):
##			if phoneme[i] in vowel:
##				contain2.append(phoneme[i])
##				number_vowel = number_vowel + 1
##		if len(word2) < 3:
##			if len(word2) == 1:
##				word_suffix = english.index(word2[0]) + 1
##			elif len(word2) == 2:
##				word_suffix = (english.index(word2[0]) + 1) * 100 + english.index(word2[1]) + 1
##		else:
##			#			word_suffix = ((english.index(newword[-3]) + 1) * 100 + (english.index(newword[-2]) + 1)) * 100 + english.index(newword[-1])+1
##
##			word_suffix = (english.index(word2[-2]) + 1) * 100 + english.index(word2[-1]) + 1
##
##		# get the last feature
##		if number_vowel == 1:
##			last = vowel.index(contain2[0]) + 1
##		else:
##			last = (vowel.index(contain2[-2]) + 1) * 100
##			last = last + vowel.index(contain2[-1]) + 1
##
##		# get the last feature
###		if number_vowel == 1:
###			last = '1' + str(vowel.index(contain2[-1]) + 50)
###		elif number_vowel == 2:
###			last = '2' + str(vowel.index(contain2[-2]) + 50) + str(vowel.index(contain2[-1]) + 50)
###		elif number_vowel == 3:
###			last = '3' + str(vowel.index(contain2[-2]) + 50) + str(vowel.index(contain2[-1]) + 50)
###		elif number_vowel == 4:
###			last = '4' + str(vowel.index(contain2[-2]) + 50) + str(vowel.index(contain2[-1]) + 50)
##
##		for j in range(len(phoneme)):
##			if phoneme[j] in vowel:
##				vowelnum = vowelnum + 1
##				location = vowel.index(phoneme[j]) + 1
##				if location < 10:
##					combination = combination[0] + '0' + str(location) + combination[1:]
##				else:
##					combination = combination[0] + str(location) + combination[1:]
##
##			else:
##				consonantnum = consonantnum + 1
##
##
##		instance.append([last, vowelnum, combination, hou])
##	x_train = pd.DataFrame(data=instance, columns=features)
####	print(x_train.head())
##
##	answer = list(clf.predict(x_train))
##
##	return answer


if __name__ == '__main__':
	training_data = helper.read_data('./asset/training_data.txt')
	classifier_path = './asset/classifier.dat'
#	train(training_data, classifier_path)
#	xx = 0
#	for i in range(30):
#		yy = train(training_data, classifier_path)
#		xx = xx + yy
#	print(xx/30)

	train(training_data, classifier_path)
##	test_data = helper.read_data('./asset/tiny_test2.txt')
##	prediction = test(test_data, classifier_path)
##	print(prediction)
##
