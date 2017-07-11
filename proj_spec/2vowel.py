import helper

training_data = helper.read_data('./asset/training_data.txt')
file = "/Users/yananzhang/Desktop/2po.txt"
vowel = ["AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"]
with open(file,"w") as f:
    for word in training_data:
        pronounce1 = word[word.index(":")+1:]
        pronounce = pronounce1.split()
        vowels = []
        vowel_num = 0
        confirm = 0
        stress = 0
        for tiny_pron1 in pronounce:
            
            if tiny_pron1[:2] in vowel:
                vowels.append(tiny_pron1)
        
        
        for tiny_pron2 in vowels:
            if "1" in tiny_pron2:
                stress = vowels.index(tiny_pron2)+1
                
        if stress == 2:
            f.write(pronounce1+"\n")
                
##        print(pronounce[0][:2])
##        if confirm == 2:
##            for tiny_pron in pronounce:
##                
##                if "0" in tiny_pron:
##                
##                    vowel_num = vowel_num+1
##                if "1" in tiny_pron:
##                    vowel_num = vowel_num+1
##                if "2" in tiny_pron:
##                    vowel_num = vowel_num+1
##            if vowel_num == 2:
##                
##                f.write(pronounce1+"\n")
##                
                    
