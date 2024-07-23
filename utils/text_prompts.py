
FRAMES = """
    1: Economic - costs, benefits, or other financial implications,
    2: Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems, 
    3: Morality - religious or ethical implications,
    4: Fairness and equality - balance or distribution of rights, responsibilities, and resources,
    5: Legality, constitutionality and jurisprudence - rights, freedoms, and authority of individuals, corporations, and government,
    6: Policy prescription and evaluation - discussion of specific policies aimed at addressing problems,
    7: Crime and punishment - effectiveness and implications of laws and their enforcement,
    8: Security and defense - threats to welfare of the individual, community, or nation,
    9: Health and safety - health care, sanitation, public safety,
    10: Quality of life - threats and opportunities for the individual's wealth, happiness, and well-being,
    11: Cultural identity - traditions, customs, or values of a social group in relation to a policy issue,
    12: Public opinion - attitudes and opinions of the general public, including polling and demographics,
    13: Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters,
    14: External regulation and reputation - international reputation or foreign policy of the U.S,
    15: Other - any coherent group of frames not covered by the above categories."""

SYS_PROMPT = f"You are an intelligent and logical journalism scholar conducting analysis of news articles. Your task is to read the article and answer the following question about the article. "

TOPIC_PROMPT = "Output the topic of the article, along with a justification for the answer. The topic should be a single word or phrase. Format your output as a json entry with the field 'justification' and 'topic'."

SUMMARY_PROMPT = "Output a brief summary of the article. The summary should be 1-2 sentences. Format your output as a json entry with the field 'summary'."

ENTITY_PROMPT = """Output the main subjects or entities in the article.
Entities are people or organisations that play a central role in the article.
For each entity, provide the name of the entity and the sentiment towards the entity communicated in the article.
Sentiment can be positive, negative or neutral. Also provide a justification for the sentiment.
Format your output as a json entry with the field 'entities'. Each entry in the 'entities' list should have the fields 'name', 'justification', 'sentiment'."""
    
FRAMING_PROMPT = f"""Framing is a way of classifying and categorizing information that allows audiences to make sense of and give meaning to the world around them (Goffman, 1974).
Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
A set of generic news frames with an id, name and description are: {FRAMES}.
Your task is to code articles for one of the above frames and provide justification for it. Format your output as a json entry with the fields 'justification', 'frame_id', 'frame_name'.
'frame_name' should be one of the above listed frames."""

POST_PROMPT = "Output only the json and no other text."

text_prompt_dict = {
    "Topic": TOPIC_PROMPT,
    "Summary": SUMMARY_PROMPT,
    "Entity": ENTITY_PROMPT,
    "Framing": FRAMING_PROMPT,
}