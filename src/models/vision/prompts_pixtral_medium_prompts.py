prefix_instruction = """

USER: You are an intelligent and logical assistant. Your job is to see the image and then read the question. You need to answer the question based on the image. If the answer could not be answered using just the image, you should put it as "None.
"""

prompt_caption = """

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

prompt_actor = """

In the shown image, what/who is the main actor? Actors are identifiable individuals, collectives, or institutions, usually mentioned by name, which are not only subject of the news image but who are given the opportunity - via direct or indirect speech - to communicate their point of view. Actors can be persons, groups, committees, organizations, or institutions. Journalists can be coded as actors as well when they not merely act as chronicler of events and statements but add context, interpretation and/or evaluation to the article, indicated by a statement which is not solely based or does not merely sum up interpretations and/or evaluations by other quoted actors. 

Also what is the perceivable gender in the image? Perceivable gender is the way others view a person along a continuum from masculine to feminine. It is based on the person's appearance, behavior, and other characteristics.

<image>\n And now for the given the image, provide the main actor in the image, the sentiment with which the actor is portrayed in the image, and a justification for the sentiment. Sentiment can be positive, negative or neutral. If the main actor is a person, output the facial expression of the person, justify the facial expression and their perceivable gender. If it is not a person, output facial expression as "None" and "perceivable gender" as "None".

Write it in json format with fields as "main-actor", "sentiment", sentiment-justification", "facial-expression", "facial-expression-justification" and "perceivable-gender", "perceivable-gender-justification".

\nASSISTANT:
"""

prompt_symbolic = """

In the shown image, what symbolic meaning is the image trying to convey. Symbolic meanings is the hidden meaning in the image not explicitly shown.

<format>
The format of the output should be as a json file that looks follows:
{
    "symbolic-object": "<symbolic thing>",
    "symbolic-meaning": "<symbolic meaning in one or two words>",
    "symbolic-meaning-explanation": "<explanation>"
}

</format>

<image>\n And now for the image you see, what symbolic meaning is the image trying to convey. 

\nASSISTANT:
"""


# prompt_3 ="""
# <image>\n For the image shown, what is the facial expression of the main subject in the image? Subject should be visible. Say 'None' if none exists. Write it in json format with fields as "main subject", "facial expression" and "explanation".
 
# \nASSISTANT:
# """


FRAMES = f"""
    1: Economic - costs, benefits, or other financial implications. The image can includes things including but not limited to  money, funding, taxes, bank, money, meetings with a logo of a financial institution. 

    2: Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems.In the image, we can see things including but not limited to a geographical area, labour, people working in an institution, or images that convey scarcity or surplus in some way, 
    
    3: Morality - religious or ethical implications. In the image, we can see things including but not limited to god, death, protests related to humanitarian issues,
    
    4: Fairness and equality - balance or distribution of rights, responsibilities, and resources. In the image, we can see things including but not limited to the fight for LGBTQ, reproductive rights, call for stopping discrimination,
    
    5: Legality, constitutionality and jurisprudence - rights, freedoms, and authority of individuals, corporations, and government. In the image, we can see things including but not limited to judges, prisons, laws, discussions on policy,
    
    6: Policy prescription and evaluation - discussion of specific policies aimed at addressing problems. In the image, we can see things including but not limited to discussions on rule, rule making bodies, tax forms,
    
    7: Crime and punishment - effectiveness and implications of laws and their enforcement. In the image, we can see things including but not limited to criminal activities, violence, and police presence,
    
    8: Security and defense - threats to welfare of the individual, community, or nation. In the image, we can see things including but not limited to military uniforms, defense personnel, border patrol, war,
    
    9: Health and safety - health care, sanitation, public safety. In the image, we can see things including but not limited to doctors, nurses, injury, disease,
    
    10: Quality of life - threats and opportunities for the individual's wealth, happiness, and well-being. In the image, we can see things including but not limited to happiness, joy, hardships of people, homelessness,
    
    11: Cultural identity - traditions, customs, or values of a social group in relation to a policy issue. In the image, we can see things including but not limited to concerts, cultural dance, art, and prominent people related to these topics,
    
    12: Public opinion - attitudes and opinions of the general public, including polling and demographics. Includes protests, riots, and strikes (incl. hunger strikes) and inclding but not limited to Sharing petitions and encouraging people to take political action,
    
    13: Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters. In the image, we can see things including but not limited to politicians, elections, voting, political campaigns,
    
    14: External regulation and reputation - international reputation or foreign policy. In the image, we can see things including but not limited to international organizations, global discussions, cross country forums,
    
    15: None - no frame could be identified because of lack of information in the image.
    """

SYS_PROMPT = f"""USER: You are an intelligent and logical journalism scholar conducting analysis of news articles. Your task is to read the article and answer the following question about the article. """

FRAMING_PROMPT = f"""
Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
A set of generic news frames with an id, name and description are: {FRAMES}."""

POST_PROMPT_FIX = """
For images: 
It is okay (and expected) for you to use your intuition or the symbolic meaning behind the image rather than only looking at it objectively. 
However, if the image has some very minor element in the image like a photo album in the background, this can be ignored. We are interested in the larger communicative intent that gets conveyed when looking at the image, rather than minor background details..

"""
# Also note the difference between morality and fairness. Morality vs Fairness 
# 1) Morality/Humanitarian → social responsibility whhich talks about Organizations and individuals have obligation to act for the benefit of society, People have obligations to help their larger society of people, even if they don’t know them personally, Responsibilities that we owe to society/each other, Moral obligation to keep families together, protect children

# 2) Fairness/Discrimination → social justice which talks about Idea that people who lack certain rights/opportunities/status or are victims of injustice are owed remedy/restoration by the larger society, All people deserve and should have access to the same rights and resources.

PROMPT_TASK = f"""
Your task is to see the image and based on the understanding of the image, choose the correct frame.

<image>\n And now for the image you see, identify the symbolic meaning and actors used to evoke a latent message for news media users. Using this information, what frame is present in the image? Look at all the frames first and then choose the correct frame. {POST_PROMPT_FIX} Write it in json format with fields as "frame-id", "frame-name" and "frame-jusitification".

\nASSISTANT:
"""

OPEN_ENDED_PROMPT_TASK = f"""
Your task is to see the image and based on the understanding of the image, choose the correct frame.

<image>\n And now for the image you see, identify the symbolic meaning and actors used to evoke a latent message for news media users. Using this information, what frame is present in the image? Look at all the frames first and then choose the correct frame. {POST_PROMPT_FIX}
You are allowed to explain the reasoning before choosing the answer. Write it in json format with field as "frame-answer-with-reason".

\nASSISTANT:
"""

# framing prompt
prompt_generic = SYS_PROMPT + FRAMING_PROMPT + PROMPT_TASK

prompt_open_ended = SYS_PROMPT + FRAMING_PROMPT + OPEN_ENDED_PROMPT_TASK

# PROMPT_LIST_LLAVA = [prefix_instruction + prompt for prompt in PROMPT_LLAVA]

PROMPT_DICT_PIXTRAL = {'caption': prompt_caption, 'actor': prompt_actor, 'symbolic': prompt_symbolic, 'generic-frame': prompt_generic, 
'open-ended-frame': prompt_open_ended}

PROMPT_DICT_PIXTRAL_MEDIUM = {k: prefix_instruction + v for k, v in PROMPT_DICT_PIXTRAL.items()}