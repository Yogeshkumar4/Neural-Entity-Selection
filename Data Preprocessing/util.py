import sys, io, re

feature_sentences = io.open("./TestSet/src-test-features.txt", "r", encoding="utf-8-sig")

f = open("test_sentences.txt","w")
max1 = 0
for line in feature_sentences:
	flag = False
	word_list = []
	for k in line.split(" "):
		this_word = ""
		for elem in k:
			if ord(elem) == 65512:
				break
			else:
				this_word = this_word + elem
		word_list.append(this_word)
	max1 = max(max1, len(word_list))	
	f.write(' '.join(word_list).encode('utf-8'))
	f.write('\n')
print(max1)	
		