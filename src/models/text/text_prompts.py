
FRAMES = """
A list of frame names and their descriptions used in news is:
    Economic - costs, benefits, or other financial implications,
    Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems, 
    Morality - religious or ethical implications,
    Fairness and equality - balance or distribution of rights, responsibilities, and resources,
    Legality, constitutionality and jurispudence - rights, freedoms, and authority of individuals, corporations, and government,
    Policy prescription and evaluation - discussion of specific policies aimed at addressing problems,
    Crime and punishment - effectiveness and implications of laws and their enforcement,
    Security and defense - threats to welfare of the individual, community, or nation,
    Health and safety - health care, sanitation, public safety,
    Quality of life - threats and opportunities for the individual's wealth, happiness, and well-being,
    Cultural identity - traditions, customs, or values of a social group in relation to a policy issue,
    Public Opinion - attitudes and opinions of the general public, including polling and demographics,
    Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters,
    External regulation and reputation - international reputation or foreign policy of the U.S,
    None - none of the above or any frame not covered by the above categories."""

SYS_PROMPT = f"You are an intelligent and logical journalism scholar conducting analysis of news articles. Your task is to read the article and answer the following question about the article. Only output the json and no other text.\n"

TOPIC_PROMPT = "Output the topic of the article, along with a justification for the answer. The topic should be a single word or phrase. Format your output as a json entry with the field 'topic_justification' and 'topic'."

SUMMARY_PROMPT = "Output a brief summary of the article. The summary should be 1-2 sentences. Format your output as a json entry with the field 'summary'."

ENTITY_PROMPT = """Your task is to identify the main subject or entity in the article that is central to the article. Entities are people or organizations that play a central role. If there are multiple entities being discussed, choose the one that is central to the article. If there are no central or clearly identifiable entities, choose "None".
Additionally, analyse the image and output the sentiment with which the subject is portrayed in the image. The sentiment can be "positive", "negative" or "neutral". In case of no entity, output "None.
If the subject is a person, also identify the perceivable gender of the subject. The possible values are "male", "female", or "non-binary". In case the subject is not a person or the gender cannot be identified, output "None". Output only the json and no other text.
Format your output as a json entry as follows:

{"entity-name": "<name of the entity>", "entity-gender":"<gender of the entity>", "sentiment": "<sentiment towards the entity>", "sentiment-reason": "<reasoning for the portrayed sentiment>"}

For the given article, provide the name of the entity, the gender of the entity (if applicable), sentiment towards the entity, and reasoning for the chosen sentiment."""
    
GENERIC_FRAMING_PROMPT = f"""Framing is a way of classifying and categorizing information that allows audiences to make sense of and give meaning to the world around them (Goffman, 1974).
Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
A set of generic news frames with an id, name and description are: {FRAMES}.
Your task is to code articles for one of the above frames and provide justification for it. Format your output as a json entry with the fields 'frame_justification', 'frame_id', 'frame_name'.
'frame_name' should be one of the above listed frames. Only output one frame per article."""

GENERIC_FRAMING_MULTIPLE_PROMPT = """
Given the list of news frames, and the news article.
Your task is to carefully analyse the article and choose the appropriate frames used in the article from the above list.
Output your answer in a json format with the format:
{"frames-list": "[<All frame names that apply from list provided above>], "reason": "<reasoning for the frames chosen>"}.
Only choose the frames from the provided list of frames. If none of the frames apply, output "None" as the answer.
"""

TONE_PROMPT = "Output the tone of the article. The tone is one or two words that describes the communicative tone that is used by the author when writing the article. Some examples of tone are Fatalistic, Optimistic, Neutral, Passionate, Pessimistic. Provide a justification for the tone. Format your output as a json entry with the fields 'justification_tone' and 'tone'."

ISSUE_FRAMING_PROMPT = """
Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
There are several ways to cover a specific issue in the news. For instance, the issue of climate change can be framed as a scientific, a political, a moral, or a health issue etc. with issue-specific frames such as "Global Doom", "Local Tragedies", "Sustainable future".
Similarly, articles related to immigration can frame immigrants as a hero, a victim, or a threat with frames such as "Economic Burden", "Cultural Invasion", "Humanitarian Crisis".
Based on the topic of the article, come up with an issue-specific frame that is relevant to the topic of the article. Provide a justification for the frame. 
Format your output as a json entry with the fields 'issue_frame_justification' and 'issue_frame'."""

POST_PROMPT = " Output only the json and no other text. Make sure to add escape characters where necessary to make it a valid json output."

text_prompt_dict = {
    "Topic": TOPIC_PROMPT,
    "Entity": ENTITY_PROMPT,
    "GenericFraming": FRAMES+GENERIC_FRAMING_MULTIPLE_PROMPT,
    "IssueFraming": ISSUE_FRAMING_PROMPT
}