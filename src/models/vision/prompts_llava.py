prefix_instruction = """

USER: You are an intelligent and logical assistant. Your job is to see the image and then read the question. You need to answer the question based on the image. If the answer could not be answered using just the image, you should put it as "None.
"""

prompt_0 = """

Caption the shown image in detail but don't count the things you see. Describe the image in detail. Example, if you see people, dont count them, describe what they are doing, what they are wearing, etc. If you see a scene with vehicles, don't count the vehicles but rather describe the scene in detail.

<format>
The format of the output should be as a json file that looks follows:
{
    "caption": "<caption in detail>"
}
</format>


<image>\n And now caption the image in detail based on the image you see. Write it in json format with fields as "caption".

\nASSISTANT:
"""

prompt_1 = """

In the shown image, what/who is the main actor? Actors are identifiable individuals, collectives, or institutions, usually mentioned by name, which are not only subject of the news image but who are given the opportunity - via direct or indirect speech - to communicate their point of view. Actors can be persons, groups, committees, organizations, or institutions. Journalists can be coded as actors as well when they not merely act as chronicler of events and statements but add context, interpretation and/or evaluation to the article, indicated by a statement which is not solely based or does not merely sum up interpretations and/or evaluations by other quoted actors. 

Also what is the perceivable gender in the image? Perceivable gender is the way others view a person along a continuum from masculine to feminine. It is based on the person's appearance, behavior, and other characteristics.

<image>\n And now for the given the image, provide the main actor in the image, the sentiment with which the actor is portrayed in the image, and a justification for the sentiment. Sentiment can be positive, negative or neutral. If the main actor is a person, output the facial expression of the person, justify the facial expression and their perceivable gender. If it is not a person, output facial expression as "None" and "perceivable gender" as "None".

Write it in json format with fields as "main_actor", "sentiment", sentiment_justification", "facial_expression", "facial_expression_justification" and "perceivable_gender", "perceivable_gender_justification".

\nASSISTANT:
"""

prompt_2 = """

In the shown image, what symbolic meaning image is trying to convey. Symbolic meaning are hidden meaning in the images not explicitly shown in the images.

<format>
The format of the output should be as a json file that looks follows:
{
    "symbolic thing": "<symbolic thing>",
    "symbolic meaning": "<symbolic meaning in one word>",
    "explanation": "<explanation>"

}

</format>

<examples>

{
    "symbolic thing": "color red",
    "symbolic meaning": "danger",
    "explanation": "The color red in the image symbolizes danger and warns the viewer to be cautious."
}

{   
    "symbolic thing": "raised fist",
    "symbolic meaning": "power",
    "explanation": "The raised fist in the image symbolizes power and strength, suggesting that the person is fighting for their rights."
}

</examples>

<image>\n And now for the image you see, what symbolic meaning is the image trying to convey. 

\nASSISTANT:
"""


# prompt_3 ="""
# <image>\n For the image shown, what is the facial expression of the main subject in the image? Subject should be visible. Say 'None' if none exists. Write it in json format with fields as "main subject", "facial expression" and "explanation".
 
# \nASSISTANT:
# """


FRAMES = f"""
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
    14: External regulation and reputation - international reputation or foreign policy,
    15: None - no frame could be identified because of lack of information in the image."""

    # 15: Other - frame is present and can be any coherent group of frames not covered by the above categories.

SYS_PROMPT = f"""USER: You are an intelligent and logical journalism scholar conducting analysis of news articles. Your task is to read the article and answer the following question about the article. """

FRAMING_PROMPT = f"""Framing is a way of classifying and categorizing information that allows audiences to make sense of and give meaning to the world around them (Goffman, 1974).
Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
A set of generic news frames with an id, name and description are: {FRAMES}."""

PROMPT_TASK = """
Your task is to see the image and based on the understanding of the image, choose the correct frame.

<image>\n And now for the image you see, what frame is present in the image? Look at all the frames first and then choose the correct frame. Write it in json format with fields as "frame_id", "frame_name" and "frame_jusitification".

\nASSISTANT:
"""

# framing prompt
prompt_3 = SYS_PROMPT + FRAMING_PROMPT + PROMPT_TASK


# add prefix_instruction to all the prompts
PROMPT_LLAVA = [prompt_0, prompt_1, prompt_2, prompt_3]

PROMPT_LIST_LLAVA = [prefix_instruction + prompt for prompt in PROMPT_LLAVA]
