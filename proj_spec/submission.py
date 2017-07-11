## import modules here 
import helper
from sklearn import tree, model_selection
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize
################# training #################
##nltk.download('averaged_perceptron_tagger')
##def total_prefix(word):
##    suffixes=('THELESS', 'TATIVES', 'MINATE', 'UNDERREPORT', 'COMEDIENNE', 'DULATE', 'JAHIDEEN', 'NERATE', 'NASIONAL', 'SCRIBE', 'ACKED', 'ANCED', 'MINES', 
##          'NEERS', 'NOSED', 'POSED', 'SENTS', 'TANDS', 'TIANE', 'UILLE', 'VERSE', 'ADOR', 'ELLE', 'ENDS', 'ETTE', 'EURS', 'EVAN', 'IBED', 'LIED', 'MAIN', 'NECT', 'NEER', 'NOSE', 
##          'PATH', 'PPLY', 'TECT', 'TUNE', 'WEES', 'DAD', 'ERU', 'ETE', 'EUR', 'JAN', 'LET', 'MAR', 'TIF', 'TIK', 'YOR')
##
##    prefixes=('COUNTER', 'INTER', 'NITRO', 'UNDER', 'AERO', 'DEMO', 'IDIO', 'OVER', 'SOCI', 'TELE', 'BIO', 'COM', 'LAV', 'LEG', 'MIS', 'NAI', 'NAT')
##
##
##    spell = word[:word.index(":")]
##    spell = spell.lower()
##    pref = 0
##    sufx = 0
##    for pre in range(0,len(prefixes)):
##        if spell.startswith(prefixes[pre]):
##            pref = pre+1
##    for suf in range(0,len(suffixes)):
##        if spell.endswith(suffixes[suf]):
##            sufx = suf+1
##            
##    return pref,sufx

##def first_vowel(pronounce):
##    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
##    first = 0
##    for pro in pronounce:
##        if pro[:2] in vowel:
##            first = vowel.index(pro[:2])+1
##            break
##    return first
##def second_vowel(pronounce):
##    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
##    second = 0
##    time = 0
##    for pro in pronounce:
##        if pro[:2] in vowel:
##            if time == 0:
##                time = time+1
##                continue
##            if time == 1:
##                time = time+1
##                second = vowel.index(pro[:2])+1
##                break
##            else:
##                break
##    return second
##def third_vowel(pronounce):
##    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
##    third = 0
##    time = 0
##    for pro in pronounce:
##        if pro[:2] in vowel:
##            if time < 2:
##                time = time+1
##                continue
##            if time == 2:
##                time = time+1
##                third = vowel.index(pro[:2])+1
##                break
##            else:
##                break
##    return third
##def forth_vowel(pronounce):
##    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
##    forth = 0
##    time = 0
##    for pro in pronounce:
##        if pro[:2] in vowel:
##            if time < 3:
##                time = time+1
##                continue
##            if time == 3:
##                time = time+1
##                forth = vowel.index(pro[:2])+1
##                break
##            else:
##                break
##    return forth

def vowel_num(pronounce):
    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
    vowel_num = 0
    word_vowel = []
    for tiny_pro in pronounce:
        if tiny_pro[:2] in vowel:
            vowel_num = vowel_num+1
            word_vowel.append(tiny_pro)
            
    return vowel_num,word_vowel
def combine(pronounce):
    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
    com = ""
    comp = []
    for tiny in pronounce:
        if tiny[:2] in vowel:
            comp.append(vowel.index(tiny[:2])+1)
    
    comp.reverse()
    la = 0
    for vo in comp:
        la = la*100+vo
##    la = ((english_letter.index(spell[-3])+1)*100 +(english_letter.index(spell[-2])+1))*100+english_letter.index(spell[-1])+1
##    for aa in comp:
##        if aa >9:
##            com = com+str(aa)
##        else:
##            aa = "0"+str(aa)
##            com = com+aa

    return la
##def typeword(word):
##    
##    total_type = ['RB','JJ','FW','NNP','JJR','CC','WP','WDT','VB','NN','NNS','IN','DT']
##    spell = word[:word.index(":")]
##    
##    tagged = nltk.pos_tag([spell])
##    
##    type_word = total_type.index(tagged[0][1])+1
##    return type_word

##def two_prefix(word):
##    total = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","P","B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","R","S","SH","T","TH","V","W","Y","Z","ZH"]
##    two_pronounce = word[word.index(":")+1:]
##    two_pronounce = two_pronounce.split()
##    first_prefix = 0
##    second_prefix = 0
##    first_prefix = total.index(two_pronounce[0][:2])+1
##    second_prefix = total.index(two_pronounce[1][:2])+1
##
##    return first_prefix,second_prefix
def two_vowel(word):
    total = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
    two_pronounce = word[word.index(":")+1:]
    two_pronounce = two_pronounce.split()
    last_vowel = 0
    second_last_vowel = 0
    tiny_vow = []
    for ti in two_pronounce:
        if ti[:2] in total:
            tiny_vow.append(total.index(ti[:2])+1)
    last_vowel = tiny_vow[-1]
    second_last_vowel = tiny_vow[-2]*100
##    if last_vowel < 10:
##        last_vowel = "0"+str(last_vowel)
##    else:
##        last_vowel = str(last_vowel)
##    if second_last_vowel < 10:
##        second_last_vowel = "0" +str(second_last_vowel)
##    else:
##        second_last_vowel = str(second_last_vowel)
##    print(second_last_vowel+last_vowel)
    return second_last_vowel+last_vowel
##    second_prefix = total.index(two_pronounce[1][:2])+1
##
##    return first_prefix,second_prefix
def last(word,last_dic):
    spell = word[:word.index(":")]
    if spell[-1] == "S" or spell[-1] == "D":
        spell = spell[:-1]
    english_letter = ["A","B","C","D","E","F","G","H","I","G","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    if len(spell) < 3:
        la = 0
    else:
        if spell[-3:] in last_dic:
            
            la = ((english_letter.index(spell[-3])+1)*100 +(english_letter.index(spell[-2])+1))*100+english_letter.index(spell[-1])+1
        else:
            la = 0
    return la
def three_letter(data):
    last_dic = {}
    dic_keys = []
    for word in data: 
       spell = word[:word.index(":")]
       if spell[-1] == "S" or spell[-1] == "D":
           spell = spell[:-1]
       lastword = spell[-3:]
       if lastword in last_dic.keys():
           last_dic[lastword] += 1          
       else:
           last_dic[lastword] = 1
    for ke in last_dic:
        dic_keys.append(ke)
    
    for clear in dic_keys:
        if last_dic[clear]<30:
            del last_dic[clear]
    
    return last_dic.keys()
    
    


def train(data, classifier_file):# do not change the heading of the function
    global last_dic
    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
    consonant = ["P","B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","R","S","SH","T","TH","V","W","Y","Z","ZH"]
    instances = []
    attributes = ["two_vowel","vowel_num","combine","last_word","position"]
    features = ["two_vowel","vowel_num","combine","last_word"]
    firstvowel = 0
    secondvowel = 0
    thirdvowel = 0
    forthvowel = 0
    last_dic = three_letter(data)
    for word in data:
        tokens = []
        word_vowel = []
        type_word = 0
        first_prefix = 0
        second_prefix = 0
        pronounce = word[word.index(":")+1:]
        
        pronounce = pronounce.split()
        ## vowel_num feature
        vowel_number,word_vowel = vowel_num(pronounce)
        ## record exact vowel
##        firstvowel = first_vowel(pronounce)
##        secondvowel = second_vowel(pronounce)
##        thirdvowel = third_vowel(pronounce)
##        forthvowel = forth_vowel(pronounce)
        ## judge lastword
        lastword = last(word,last_dic)
        
        ## combine
        comb = combine(pronounce)
        ## generate last two vowels combine
        vowel_com = two_vowel(word)
        ## judge prefix
##        prefix,suffix = total_prefix(word)
       
        ## judge the type of the word
##        type_word = typeword(word)
##            first_prefix,second_prefix = two_prefix(word)
##            print(word)
##            print(type_word)
##            print(first_prefix)
##            print(second_prefix)
            
            
        ##get the stress position
        for tiny_pron in word_vowel:
            if "1" in tiny_pron:               
                stress = word_vowel.index(tiny_pron)+1
##                print(stress)
##                print(pronounce)
        
        instances.append([vowel_com,vowel_number,comb,lastword,stress])
    
    df = pd.DataFrame(data=instances,columns = attributes)
    x = df[features]
    y = df.position    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.2)
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
       
    model=clf.fit(x, y)
##    prediction = list(clf.predict(x_test))
##    ground_truth = list(y_test)
##    precision = [0,0,0,0]
##    count = [0,0,0,0]
##    for i in range(len(ground_truth)):
##        if ground_truth[i] == prediction[i]: 
##           precision[ground_truth[i] - 1] +=1
##        count[ground_truth[i] - 1] += 1
##    for j in range(len(count)):
##        precision[j] = precision[j]/count[j]
##    print(count)
##    print(precision)
##    print(f1_score(ground_truth,prediction,average = 'macro'))
    output = open(classifier_file,'wb')
    pickle.dump(clf,output)
    output.close()




################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    global last_dic
    clf = pickle.load(open(classifier_file,"rb"))
    vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
    consonant = ["P","B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","R","S","SH","T","TH","V","W","Y","Z","ZH"]
    instances = []
    attributes = ["two_vowel","vowel_num","combine","last_word","position"]
    features = ["two_vowel","vowel_num","combine","last_word"]
    
    for word in data:
        tokens = []
        word_vowel = []
        type_word = 0
        first_prefix = 0
        second_prefix = 0
        pronounce = word[word.index(":")+1:]
        
        pronounce = pronounce.split()
        ## vowel_num feature
        vowel_number,word_vowel = vowel_num(pronounce)
        ## judge lastword
        lastword = last(word,last_dic)
##        print(lastword)
        ## combine
        comb = combine(pronounce)
        ## generate last two vowels combine
        vowel_com = two_vowel(word)
        ## judge prefix
##        prefix,suffix = total_prefix(word)
        ## type of the word
##        type_word = typeword(word)
       
        instances.append([vowel_com,vowel_number,comb,lastword])
    df = pd.DataFrame(data=instances,columns = features)
    x = df[features]
    
    prediction = list(clf.predict(x))
    return prediction
  

    
##
##if __name__ == '__main__':
##    
##    training_data = helper.read_data('./asset/training_data.txt')
##    classifier_path = './asset/classifier.dat'
##    train(training_data,classifier_path)
##    test_data = helper.read_data('./asset/tiny_test.txt')
##    prediction = test(test_data,classifier_path)
##    print(prediction) 
##	

##    
