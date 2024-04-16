def xquad(list_lib_score, list_lib_frequency, frequency_cut_off = 20, gamma = 0.2, number_cut_off_lib = 100, p_l_c = 0.3):
    xquad_score  = {}
    
    for key,value in list_lib_score.items():
        new_value = 0
        new_value += (1-gamma)*value 
        second_value = 0
        list_h_c = ['head','tail']
        score1 = 0
        score2 = 0
        for category in list_h_c:
            for lib in list_lib_score:
                if list_lib_frequency[lib] >= frequency_cut_off:
                    head += 1
                else:
                    tail += 1
            if category == 'head':
                p_c_p  = head/number_cut_off_lib
                p_l_c = p_l_c
                if not xquad_score: 
                    new_value += gamma*p_c_p*p_l_c*1
                else:                
                    multiple = 1
                    for key, value in xquad_score.items():
                        # head
                        if list_lib_frequency[lib] >= frequency_cut_off:
                            multiple = 0 
                            break
                        else:
                            multiple = 1
                    new_value += gamma*p_c_p*p_l_c*multiple
            else:
                p_c_p = tail/number_cut_off_lib
                p_l_c = 1 - p_l_c
                if not xquad_score: 
                    new_value += gamma*p_c_p*p_l_c*1
                else:                
                    multiple = 1
                    for key, value in xquad_score.items():
                        # head
                        if list_lib_frequency[lib] < frequency_cut_off:
                            multiple = 0 
                            break
                        else:
                            multiple = 1
                    new_value += gamma*p_c_p*p_l_c*multiple
            
            
            xquad_score.append({key : new_value})

    
    # p(l|p) is score 
    # p(c|p) is probabilty l is on category c , ex: 7 head, 3 tail => 0,3 0,7
    # p(l|c) la ty le head va tail
    # p(i|c,S) is set to 1 if item i belongs to both S and category c, and 0 otherwise.
    # (1 - p(i|c,S)) is set to 1 if item i belongs to both S and category c, and 0 otherwise. 
    
    
from gensim.models import Word2Vec
import os

def read_files(folder_path):
    paragraphs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        x = [] 
        paragraph = []
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip().split()
                    x.append(line[1])                    
                paragraphs.append(' '.join(x[1:]))
    return paragraphs


folder_path = '/users/anhld/BiasInRSSE/CROSSREC/D2'
paragraphs = read_files(folder_path)
print(paragraphs[0])