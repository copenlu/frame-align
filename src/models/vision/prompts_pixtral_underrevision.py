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
    1: Economic - costs, benefits, or other finance related. The image can includes things including but not limited to  money, funding, taxes, bank, meetings with a logo of a financial institution. If you are using logo of a financial instituion to classify it as economic, make sure it is clearly visible. If it is not clearly visible, it should be classified as 'None'. A professional attire in itself doesn not mean economic frame. 

    2: Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems. In the image, we can see things including but not limited to a geographical area, farmland, agriculture land, labour, people working in an institution, or images that convey scarcity or surplus in some way. 
    
    3: Morality - religious or ethical implications. In the image, we can see things including but not limited to god, death, priests, church, protests related to moral issues.
    
    4: Fairness and equality - balance or distribution of rights, responsibilities, and resources. In the image, we can see things including but not limited to the fight for civil or political rights, LGBTQ,  or calls to stopping discrimination.
    
    5: Legality, constitutionality and jurisprudence - legal rights, freedoms, and authority of individuals, corporations, and government. In the image, we can see things including but not limited to , prisons, laws, judges in robes, courtrooms, legal documents, and prison facilities. This does not include sports contexts, such as referees or players enforcing or breaking game rules.
    
    6: Policy prescription and evaluation - discussion of specific policies aimed at addressing problems. In the image, we can see things including but not limited to discussions on rule, rule making bodies, people in formal settings—such as boardrooms or legislative halls—actively debating, analyzing data, and reviewing policy drafts or proposals. You might see charts, graphs, or official documents. People in formal attire with no other information should not be classified as policy prescription and evaluation.
    
    7: Crime and punishment - effectiveness and implications of laws and their enforcement. In the image, we can see things including but not limited to criminal activities, violence, police officers making arrests, crime scenes with investigators, courtrooms during criminal trials, prisons and detainees. This frame specifically excludes contexts involving sports, such as referees, players, or rule enforcement within games, which are not related to societal law violations or legal punishment.
    
    8: Security and defense - threats to the individual, community, or nation. In the image, we can see things including but not limited to military uniforms, defense personnel, border patrol, war, soldiers, military equipment like tanks or fighter jets, border walls, or surveillance systems monitoring wide areas.
    
    9: Health and safety - health care, sanitation, public safety. Images with objects like coffee, drinks, food items or activities like sports which a clear and literal message that it affects health and safety positively or negatively should be classified as health and safety, otherwise it should be classified as 'None'. E.g. a person drinking coffee does not mean health and safety, but a person drinking a medicine or having cigarette does. A bus does not mean health and safety, but a bus with a warning sign does.  In the image, we can see things including but not limited to doctors, nurses, injury, disease, or events with environmental impact that may impact health and safety. 
    
    10: Quality of life - threats and opportunities for the individual's wealth, happiness, and well-being. In the image, we can see things that improves happiness or demonstrates quality of life in some form. It also includes things that demonstrate deterioration of quality of life by showing hardships of people, homelessness etc. This may also include happy children, food items that demonstrate good quality of life or people enjoying a nice meal.
    
    11: Cultural identity - traditions, customs, or values of a social group in relation to a policy issue. In the image, we can see things including but not limited to concerts, cultural dance, sports, art, celebrities, artists and prominent people related to these topics. Examples, celebrities, traditional dress, sports with clear countriy specific detail e.g. jerseys/flags, culural events, cultural art etc. Otherwise, it should be classified as 'None'.
    
    12: Public opinion - attitudes and opinions of the general public, including polling and demographics. Includes generic protests, people (non-celebrities) engaging with large crowds, riots, and strikes and including but not limited to sharing petitions and encouraging people to take political action. It will also include news broadcasts, talk shows, and interviews with people that are related to public opinion at large. 
    
    13: Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters. In the image, we can see things related to politicians, elections, voting, political campaigns. Just formal clothing does not mean political frame. If the images does not have a political person which is recognizable, it should not be classified as political. A formal attire with no political information should be classified as 'None'.
    
    14: External regulation and reputation - international reputation or foreign policy. In the image, we can see things including but not limited to international organizations, global discussions/meetings, foreign policy, flags from multiple countries, or delegates at a cross-country forum discussing reputation and regulation. If you use a logo of a global organization to classify it as external regulation and reputation, make sure it is clearly visible in the image. If it is not clearly visible, it should be classified as 'None'.
    
    15: None - no frame could be identified because of lack of information in the image. This should be selected when no other frame is applicable. Example, a handshake with no other information, a logo of a company with no other information, a landscape with no other information, a person in a photo album with no other information, a person speaking with no other information about the content of the speech or person's identity, a formal event with  no other information, a person in formal attire with no other information, a news logo with no news, simple objects like vehicle/car/pen/paper/sign-boards/objects etc with no other information etc.
    """

SYS_PROMPT = f"""You are an intelligent and logical journalism scholar conducting analysis of images associated with news artciles."""

# FRAMING_PROMPT = f
# Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
# Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
FRAMING_PROMPT = f"""A set of generic news frames with an id, name and description are: {FRAMES}."""

POST_PROMPT_FIX = """
If the image has some minor elements like a photo album or painting in the background or the clothing of the subject, this should be ignored as it does not capture the larger communicative intent. For every case where an image does not clearly fit into one of the frames, you should select "None". Images related to sports which have specific information of countries should be classified as 'cultural identity' otherwise 'None'. A competitive sports with no country detail or an image of a referee is not crime and punishment frame because regulated sports are not criminal activities; it will be 'None' frame. Similarly, if the image is simply of a handshake with no other information, it will be 'None'. An image of a person in formal attire with no other information about the person or the event is always classified as 'None'. Do not make symbolic or metaphorical interpretations which are not clearly visible in the image. E.g. a grassy field or empty pathway should not be used metaphorically. It can be 'capacity and resources' if it is a farm/agricultural field or 'None' if it is just an empty field/pathway with no other information."""


PROMPT_TASK = f"""
Your task is to see the image and based on the understanding of the image, choose the correct frame from the above list.

<image>

And now for the image you see, look at frames list provided first and then choose the correct frame from the list of provided frames. Identify the frame name associated with the larger theme that can evoke a latent message for readers of the article where this image is present. Using this information, identify the frame present in the image. Use only what is literally seen in the image to classify it. If there is text in image, extract it and use the information fo framing if it contributes to a frame. Your output should be in a json format with the fields "frame-id", "frame-name" and "frame-jusitification". {POST_PROMPT_FIX} Only identify the frame that is most relevant to the image. The frame has to be one provided in the list. If there is no frame from the list that is clearly communicated, choose "None".

ASSISTANT:"""


OPEN_ENDED_PROMPT_TASK = f"""
Your task is to see the image and based on the understanding of the image, choose the correct frame.

<image>\n And now for the image you see, identify the symbolic meaning and actors used to evoke a latent message for news media users. Using this information, what frame is present in the image? Look at all the frames first and then choose the correct frame. {POST_PROMPT_FIX}
You are allowed to explain the reasoning before choosing the answer. Write it in json format with field as "frame-answer-with-reason".

\nASSISTANT:
"""

# framing prompt
prompt_generic =  SYS_PROMPT + FRAMING_PROMPT + PROMPT_TASK

# prompt_open_ended = FRAMING_PROMPT + OPEN_ENDED_PROMPT_TASK

# PROMPT_LIST_LLAVA = [prefix_instruction + prompt for prompt in PROMPT_LLAVA]

# PROMPT_DICT_PIXTRAL = {'caption': prompt_caption, 'actor': prompt_actor, 'symbolic': prompt_symbolic, 'generic-frame': prompt_generic, 
# 'open-ended-frame': prompt_open_ended}

PROMPT_DICT_PIXTRAL = {'generic-frame': prompt_generic}

PROMPT_DICT_PIXTRAL_REVISION = {k: prefix_instruction + v for k, v in PROMPT_DICT_PIXTRAL.items()}



# For instance, an image should only be termed as political when the subjects are clearly politicians. If the image is of a person in formal attire, it should not be classified as political unless there is other information present in the image that clearly indicates a political frame. Similarly, if the image is simply of a handshake with no other information, it will be 'None'. If handshake is with image of a bank, it could be economic; in a press conference it could mean External regulation and reputation.  Images related to sports which have specific information of countries should be classified as 'cultural identity' otherwise 'None'. 


# For e.g., a handshake doesnot mean economic tranasction. Image showing happy children shows quality of life or cultural identiyy rather than threat or security. lanscape or open public space might be related to quality of life or capacity and resources. If it is a logo of a company, classify it as a none. Formal attire does not necesarily mean political frame, it can be cultural identity as well or may be none if no other information is present.
# Also note the difference between morality and fairness. Morality vs Fairness 
# 1) Morality/Humanitarian → social responsibility whhich talks about Organizations and individuals have obligation to act for the benefit of society, People have obligations to help their larger society of people, even if they don’t know them personally, Responsibilities that we owe to society/each other, Moral obligation to keep families together, protect children

# 2) Fairness/Discrimination → social justice which talks about Idea that people who lack certain rights/opportunities/status or are victims of injustice are owed remedy/restoration by the larger society, All people deserve and should have access to the same rights and resources.