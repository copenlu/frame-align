TOPIC_LABEL_PROMPT = """
                    Below is a list of topics for a set of US based news articles.
                    Your job is to come up high level categories for each of them like Business, Politics, Legal, Immigration, War, Sports etc.
                    If the topic is not in English, is gibberish, or you cannot get a broad topic for it, output no_topic. Also dont put multiple topics like War/Crime, just put one of them.
                    Output a dictionary with topic as the key and high level category as the value, in the same order. Output no other tokens, only the dictionary. 
                    List of topics:
                    """

# MERGE_TOPICS_PROMPT = """Below is a list of broad topics extracted from US news based articles. 
#                     However, there is some repetitions in them.
#                     Merge the repeated topics into one with the appropriate name and output a dictionary with 
#                     broad topics with repetitions as keys and new topic as value.
#                     List each topic as a separate key.
#                     You do not need to come with even broader categories, just merge the almost duplicated ones.
#                     Output no other tokens, only the dictionary. List:"""