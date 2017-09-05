import sys, io, re
from difflib import SequenceMatcher


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results 

feature_sentences = io.open("./src-train-features.txt", "r", encoding="utf-8-sig")
answers_dev1 = open("./ans_train1_tok.tsv", "r").readlines()
answers_dev2 = open("./ans_train2_tok.tsv", "r").readlines()
answers_dev3 = open("./ans_train3_tok.tsv", "r").readlines()
counter_checker_extra = open("./question_train_tok.tsv", "r").readlines()
counter_checker_extra = list(map(str.rstrip,counter_checker_extra))
counter_checker_extra = list(map(str.lower, counter_checker_extra))
counter_checker_trim = open("./tgt-train.txt", "r").readlines()
f = open("NES_Dataset.txt", "w")
counter_ans = counter_checker_extra.index(counter_checker_trim[0].rstrip())
curr_index = 0
for line in feature_sentences:
	word_list = line.split(" ")
	word_list = list(filter(None, word_list))
	curr_ner = ""
	modified_line = ""
	num = 0
	ner_list = []
	my_sent = []
	for k in word_list:
		our_ner = ""
		this_ner = ""
		counter = 0
		for elem in k:
			if ord(elem) == 65512:
				counter = counter + 1
			elif counter == 2:
				our_ner = our_ner + elem
			elif counter == 0:
				this_ner = this_ner + elem
		my_sent.append(this_ner)		
		if our_ner == 'O':
			curr_ner = 'O'
		elif curr_ner == our_ner:
			ner_list[-1] = ner_list[-1] + " " + this_ner
		else:
			curr_ner = our_ner
			num += 1
			ner_list.append(this_ner)	
	ans1 = ' '.join(list(map(str.strip, answers_dev1[counter_ans].lower().split(" "))))
	ans2 = ' '.join(list(map(str.strip, answers_dev2[counter_ans].lower().split(" "))))
	ans3 = ' '.join(list(map(str.strip, answers_dev3[counter_ans].lower().split(" "))))
	if len(ner_list) > 0:		
		for elem in ner_list:	
			for x in find_sub_list(elem.split(" "),my_sent):
				f.write(' '.join(my_sent).rstrip() + "@~@")
				f.write(str(x[0]) + "," + str(x[1]) + "@~@")
				if SequenceMatcher(None,elem,ans1).ratio() > 0.5 or SequenceMatcher(None,elem,ans2).ratio() > 0.5 or SequenceMatcher(None,elem,ans3).ratio() > 0.5 or elem in ans1 or elem in ans2 or elem in ans3:
					f.write("1\n")
				else:
					f.write("0\n")
	curr_index = curr_index + 1
	try:
		counter_ans = counter_checker_extra.index(counter_checker_trim[curr_index].rstrip())			
	except:
		print(curr_index)