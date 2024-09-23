TOPIC_LABEL_PROMPT = """
                    Below is a list of underscore separated keywords related to a topic output by BERTopic when analyzing a set of US based news articles.
                    Your job is to come up high level categories for each of them like Business, Politics, Sports etc.
                    If the set of keywords is not in English, is gibberish, or you cannot get a broad topic for them, output no_topic.
                    Output a dictionary with the keywords as keys and broad topics for the corresponding keywords as values, in the same order. Output no other tokens, only the dictionary. List:
                    """

MERGE_TOPICS_PROMPT = """Below is a list of broad topics extracted from US news based articles. 
                    However, there is some repetitions in them.
                    Merge the repeated topics into one with the appropriate name and output a dictionary with 
                    broad topics with repetitions as keys and new topic as value.
                    List each topic as a separate key.
                    You do not need to come with even broader categories, just merge the almost duplicated ones.
                    Output no other tokens, only the dictionary. List:"""