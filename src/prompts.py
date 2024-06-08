########### CREATIVITY ###########
PROMPT_CREATIVITY = '''
How creative is this visual advertisement? Give your answer in the scale of 1 to 5 with 1 being not creative at all, 3 being neutral, and 5 being very creative.
'''.strip()

# PROMPT_CREATIVITY_PARSING = '''
# User response: {vlm_output}
# On a scale of 1 to 5, with 1 being not creative at all, 3 being neutral, and 5 being very creative, the user's response in a single number is:
# '''.strip() # TODO

PROMPT_CREATIVITY_DISAGREEMENT = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it creative (i.e. compose of creative ideas) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score between 1 to 5 with 1 being no agreement and 5 being strong agreement.
'''.strip() 

# Give your best guess in the scale of 1 to 5 
# with 1 being no disagreement at all, 3 being neutral, and 5 being lots of disagreement.

# PROMPT_CREATIVITY_DISAGREEMENT_PARSING  = '''
# How creative is this visual advertisement? Give your answer in the scale of 1 to 5 with 1 being not creative at all, 3 being neutral, and 5 being very creative.
# '''.strip() # TODO


########### ATYPICALITY ###########
PROMPT_ATYPICALITY = '''
How unusual about the advertisement? Give your answer in the scale of 1 to 5 with 1 being very usual, 3 being neutral, and 5 being very unusual.
'''.strip()

PROMPT_ATYPICALITY_DISAGREEMENT  = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it unusual (i.e. some abnormal objects or connections) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score between 1 to 5 with 1 being no agreement and 5 being strong agreement.
'''.strip() 


########### ORIGINALITY ###########
PROMPT_ORIGINALITY = '''
How original is this visual advertisement? Give your answer in the scale of 1 to 5 with 1 being very usual, 3 being neutral, and 5 being very unusual.
'''.strip() # TODO

PROMPT_ORIGINALITY_DISAGREEMENT  = '''
I am about to ship this advertisement design to the public and I am unsure how would the audience intepret it.
Some might consider it original (i.e. unique of its kind) while some others would not. To what extent would they agree on each other? 
Make your best guess and give me an agreement score between 1 to 5 with 1 being no agreement and 5 being strong agreement.
'''.strip() 
