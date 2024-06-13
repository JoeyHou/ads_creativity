########### DISAGREEMENT ###########
# PROMPT_DISAGREEMENT_PARSING  = '''
# User response: {vlm_output}
# On a scale of 1 to 3, with 1 being no disagreement, 2 being neutral, and 3 being lots of disagreement, the user's disagreement is:
# '''.strip() # TODO

# PROMPT_PAIRWISE_PARSING = '''
# User response: {vlm_output}
# Among image on the left (image 1) and iamge on the right (image 2), which one is perferred by the user? Answer in single number:
# '''.strip() # TODO


########### CREATIVITY ###########
PROMPT_CREATIVITY = '''
How creative is this visual advertisement? Give your answer in the scale of 1 to 3 with 1 being not creative at all, 2 being neutral, and 3 being very creative. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()
# "First give the score then provide the reasoning.

# PROMPT_CREATIVITY_PARSING = '''
# User response: {vlm_output}

# On a scale of 1 to 5, with 1 being not creative at all, 2 being neutral, and 3 being very creative, the user's response in a single number is:
# '''.strip() # TODO

PROMPT_CREATIVITY_DISAGREEMENT = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it creative (i.e. compose of creative ideas) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score between 1 to 3, with 1 being easily agree (high agreement), 2 being neutral, and 3 being hardly agree (low agreement).
Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip() 

# Give your best guess in the scale of 1 to 5 
# with 1 being no disagreement at all, 3 being neutral, and 5 being lots of disagreement.
PROMPT_CREATIVITY_PAIRWISE = '''
Here are two images of advertisement. Which one is more likely to succeed in catching people's eyes by being creative? Answer 1 for the left one and 2 for the right one. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()

# PROMPT_CREATIVITY_PAIRWISE_PARSING = '''
# User Response: {vlm_output} 
# Analysis: Among image on the left (image 1) and iamge on the right (image 2), which one is more likely to succeed in catching people's eyes by being creative? 
# Answer according to User Response (in single number):
# '''.strip() # TODO










########### ATYPICALITY ###########
PROMPT_ATYPICALITY = '''
How unusual (i.e. including abnormal objects or atypical connotations) about the advertisement? Give your answer in the scale of 1 to 3 with 1 being very normal, 2 being neutral, and 3 being very unusual and abnomal. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()

# PROMPT_ATYPICALITY_PARSING = '''
# User response: {vlm_output}

# On a scale of 1 to 5, with 1 being not creative at all, 2 being neutral, and 3 being very creative, the user's response in a single number is:
# '''.strip() # TODO

PROMPT_ATYPICALITY_DISAGREEMENT  = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it unusual (i.e. some abnormal objects or connections) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score between 1 to 3, with 1 being easily agree (high agreement), 2 being neutral, and 3 being hardly agree (low agreement).
Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip() 

PROMPT_ATYPICALITY_PAIRWISE = '''
Here are two images of advertisement. Which one is more abnormal and unusual? Answer 1 for the left one and 2 for the right one. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()

# PROMPT_ATYPICALITY_PAIRWISE_PARSING = '''
# User Response: {vlm_output} 
# Analysis: Among image on the left (image 1) and iamge on the right (image 2), which one is more abnormal and unusual?
# Answer according to User Response (in single number):
# '''.strip() # TODO










########### ORIGINALITY ###########
PROMPT_ORIGINALITY = '''
How noval (i.e. unique from previous ads) is this visual advertisement? Give your answer in the scale of 1 to 3 with 1 being not original at all, 2 being neutral, and 3 being very unusual and outstanding. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip() # TODO
# PROMPT_ORIGINALITY = '''
# I am about to release this advertisement design to the public and I am unsure how would the audience intepret it.
# Can you give me an educated guess, in the scale of 1 to 5 with 1 being not original at all, 3 being neutral, and 5 being very unusual and outstanding.
# '''.strip()

# PROMPT_ORIGINALITY_PARSING = '''
# User response: {vlm_output}
# On a scale of 1 to 3, with 1 being very usual, 2 being neutral, and 3 being very unusual, the user's response in a single number is:
# '''.strip() # TODO

PROMPT_ORIGINALITY_DISAGREEMENT  = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it original (i.e. unique of its kind) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score between 1 to 3, with 1 being easily agree (high agreement), 2 being neutral, and 3 being hardly agree (low agreement).
Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip() 

PROMPT_ORIGINALITY_PAIRWISE = '''
Here are two images of advertisement. Which one is more unique compared with other ads in the same product category? Answer 1 for the left one and 2 for the right one. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()

# PROMPT_ORIGINALITY_PAIRWISE_PARSING = '''
# User Response: {vlm_output} 
# Analysis: Among image on the left (image 1) and iamge on the right (image 2), which one is more unique compared with other ads in the same product category?
# Answer according to User Response (in single number):
# '''.strip() # TODO

# mturk_prompt = {
#     'disagreement': {
#         'creativity': PROMPT_CREATIVITY_DISAGREEMENT,
#     },
#     'pairwise': {
#         'creativity': PROMPT_CREATIVITY_PAIRWISE,
#         'creativity_parsing': PROMPT_CREATIVITY_PAIRWISE_PARSING
#     },
#     'prediction': {
#         'creativity': PROMPT_CREATIVITY,
#         'creativity_parsing': PROMPT_CREATIVITY_PARSING,
#     }
# }














########### ATYPICALITY - Original Data ###########
PROMPT_ATYPICALITY_OD = '''
How unusual (i.e. including abnormal objects or atypical connotations) about the advertisement? Give an answer of either 0 or 1; answer 0 for being very normal and 1 being very unusual and abnomal. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()

PROMPT_ATYPICALITY_DISAGREEMENT_OD  = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it unusual (i.e. some abnormal objects or connections) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score of either 0 or 1, with 1 for no agreement, 0 high agreement.
Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip() 

PROMPT_ATYPICALITY_PAIRWISE_OD = '''
Here are two images of advertisement. Which one is more abnormal and unusual? Answer 1 for the left one and 2 for the right one. Give your answer in the following format: "answer: {score}; explanation: {reasoning}"
'''.strip()
