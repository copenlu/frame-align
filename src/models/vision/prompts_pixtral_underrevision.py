prefix_instruction = "USER: You are an intelligent and logical assistant. Your job is to see the image and then read the question. You need to answer the question based on the image. If the answer could not be answered using just the image, you should put it as \"None\""

prompt_entity = """
<image>\n 
Identify the main subject or entity in the image that is central to the image. Additionally, identify the gender of the subject if applicable.
Entities are people or organizations that play a central role in the image.
Provide the name of the entity, the gender of the entity (if applicable), sentiment towards the entity, and reasoning for their identification.
If there are multiple entities, choose the one that is central to the image. if there are no central entities, choose "None". If there no identifiable entities, choose "None".
Format your output as a json entry as follows:

{"entity-name": "<name of the entity>", "entity-gender":"<gender of the entity>", "gender-justification": "<reasoning for the gender identification>", "sentiment": "<sentiment towards the entity>", "reason": "<reasoning for the entity identification>"}
Output only the json and no other text.

"""

FRAMES_no_num = f"""
    Economic - costs, benefits, or other finance related. The image can includes things including but not limited to  money, funding, taxes, bank, meetings with a logo of a financial institution. If you are using logo of a financial instituion to classify it as economic, make sure it is clearly visible. If it is not clearly visible, it should be classified as 'None'. A professional attire in itself doesn not mean economic frame. 
    Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems. In the image, we can see things including but not limited to a geographical area, farmland, agriculture land, labour, people working in an institution, or images that convey scarcity or surplus in some way. 
    Morality - religious or ethical implications. In the image, we can see things including but not limited to god, death, priests, church, protests related to moral issues.
    Fairness and equality - balance or distribution of rights, responsibilities, and resources. In the image, we can see things including but not limited to the fight for civil or political rights, LGBTQ,  or calls to stopping discrimination.
    Legality, constitutionality and jurisprudence - legal rights, freedoms, and authority of individuals, corporations, and government. In the image, we can see things including but not limited to , prisons, laws, judges in robes, courtrooms, legal documents, and prison facilities. This does not include sports contexts, such as referees or players enforcing or breaking game rules.
    Policy prescription and evaluation - discussion of specific policies aimed at addressing problems. In the image, we can see things including but not limited to discussions on rule, rule making bodies, people in formal settings—such as boardrooms or legislative halls—actively debating, and reviewing policy drafts or proposals. You might see official charts, graphs, or official documents. People in formal attire with no other information should not be classified as policy prescription and evaluation.
    Crime and punishment - effectiveness and implications of laws and their enforcement. In the image, we can see things including but not limited to criminal activities, violence, police officers making arrests, crime scenes with investigators, courtrooms during criminal trials, prisons with detainees. This frame specifically excludes contexts involving sports, such as referees, players, or rule enforcement within games, which are not related to societal law violations or legal punishment.
    Security and defense - threats to the individual, community, or nation. In the image, we can see things including but not limited to military uniforms, defense personnel, border patrol, war, soldiers, military equipment like tanks or fighter jets, border walls, or surveillance systems monitoring wide areas.
    Health and safety - health care, sanitation, public safety. Images with objects like coffee, drinks, food items or activities like sports which a clear and literal message that it affects health and safety positively or negatively should be classified as health and safety, otherwise it should be classified as 'None'. E.g. a person drinking coffee does not mean health and safety, but a person drinking a medicine or having cigarette does. A bus does not mean health and safety, but a bus with a warning sign does.  In the image, we can see things including but not limited to doctors, nurses, injury, disease, or events with environmental impact that may impact health and safety. 
    Quality of life - threats and opportunities for the individual's wealth, happiness, and well-being. In the image, we can see things that improves happiness or demonstrates quality of life in some form. It also includes things that demonstrate deterioration of quality of life by showing hardships of people, homelessness etc. This may also include happy children, food items that demonstrate good quality of life or people enjoying a nice meal.
    Cultural identity - traditions, customs, or values of a social group in relation to a policy issue. In the image, we can see things including but not limited to concerts, cultural dance, sports, art, celebrities, artists and prominent people related to these topics. Examples, celebrities, traditional dress, sports with clear countriy specific detail e.g. jerseys/flags, culural events, cultural art etc. Otherwise, it should be classified as 'None'.
    Public opinion - attitudes and opinions of the general public, including polling and demographics. Includes generic protests, people (non-celebrities) engaging with large crowds, riots, and strikes and including but not limited to sharing petitions and encouraging people to take political action. It will also include news broadcasts, talk shows, and interviews with people that are related to public opinion at large. 
    Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters. In the image, we can see things related to politicians, elections, voting, political campaigns. Just formal clothing does not mean political frame. If the images does not have a political person which is recognizable, it should not be classified as political. A formal attire with no political information should be classified as 'None'.
    External regulation and reputation - international reputation or foreign policy. In the image, we can see things including but not limited to international organizations, global discussions/meetings, foreign policy, flags from multiple countries, or delegates at a cross-country forum discussing reputation and regulation. If you use a logo of a global organization to classify it as external regulation and reputation, make sure it is clearly visible in the image. If it is not clearly visible, it should be classified as 'None'.
    None - no frame could be identified because of lack of information in the image. This should be selected when no other frame is applicable. Example, a handshake with no other information, a logo of a company with no other information, a landscape with no other information, a person in a photo album with no other information, a person speaking with no other information about the content of the speech or person's identity, a formal event with  no other information, a person in formal attire with no other information, a news logo with no news, a sports event with no additional information, simple objects like vehicle/car/pen/paper/sign-boards/objects etc with no other information etc.
    """


SYS_PROMPT = f"""You are an intelligent and logical journalism scholar conducting analysis of images associated with news artciles."""

# FRAMING_PROMPT = f
# Entman (1993) has defined framing as “making some aspects of reality more salient in a text in order to promote a particular problem definition, causal interpretation, moral evaluation, and/or treatment recommendation for the item described”.
# Frames serve as metacommunicative structures that use reasoning devices such as metaphors, lexical choices, images, symbols, and actors to evoke a latent message for media users (Gamson, 1995).
FRAMING_PROMPT = "A set of generic news frames with an id, name and description are: \n"

POST_PROMPT_FIX = """
If the image has some minor elements like a photo album or painting in the background or the clothing of the subject, this should be ignored as it does not capture the larger communicative intent. For every case where an image does not clearly fit into one of the frames, you should select "None". Images related to sports which have specific information of countries should be classified as 'cultural identity' otherwise 'None'. A competitive sports with no country detail or an image of a referee is not crime and punishment frame because regulated sports are not criminal activities; it will be 'None' frame. Do not make symbolic or metaphorical interpretations which are not clearly visible in the image. If justification includes sentences like - it appears to..., it seems to be..., it can be thought of as..., it should be classified as "None"."""


MINI_MULTIPLE_FRAMES_TASK_PROMPT = """
Given the list of frames, and the image.
<image>
Your task is to carefully analyse the image and choose the appropriate frames from the above list.
Output your answer in a json format with the format:
{"frames-list": "[<All frame names that apply from list provided above>], "reason": "<reasoning for the frames chosen>"}
Output only the json and no other text.
"""

MINI_MULTIPLE_FRAMES_TASK_PROMPT_CONFIDENCE = """
Given the list of frames, and the image.
<image>
Your task is to carefully analyse the image, choose the appropriate frames from the above list, give reasoning for the frames chosen and the confidence level (i.e. either low-confidence, medium-confidence, or high-confidence).
Output your answer in a json format with the format:
{"frames-list": "[<All frame names that apply from list provided above>], "reason": "<reasoning for the frames chosen>", "confidence": "<low-confidence/medium-confidence/high-confidence>"}
The frame names have to be ones provided in the list.
If there are no frames from the list that are clearly communicated, choose "None". If you choose "None", you do not choose any other frame.Output only the json and no other text.
"""

MULTIPLE_FRAMES_TASK_PROMPT = """
Your task is to see the image and based on the understanding of the image, choose the correct frames from the above list.

<image>

And now for the image you see, look at frames list provided first and then choose the correct frames from the list of provided frames.
Identify the frame names associated with the larger theme that can evoke a latent message for readers of the article where this image is present.
Using this information, identify the frames are present in the image.
Use only what is literally seen in the image to classify it.
If there is text in the image, extract it and use the information fo framing if it contributes to a frame.
Your output should be in a json format with the fields "frame-justification" and "frame-name-list" as shown below:
{"frames-list": "[<All frame names that apply from list provided above>], "reason": "<reasoning for the frames chosen>"}
Only output the frames that are most relevant to the image.
The frame names have to be ones provided in the list.
If there are no frames from the list that are clearly communicated, choose "None".
"""

MULTIPLE_FRAMES_TASK_PROMPT_CONFIDENCE = """
Your task is to see the image and based on the understanding of the image, choose the correct frames from the above list.

<image>

And now for the image you see, look at frames list provided first and then choose the correct frames from the list of provided frames.
Identify the frame names associated with the larger theme that can evoke a latent message for readers of the article where this image is present.
Use only what is literally seen in the image to classify it.
If there is text in the image, extract it and use the information fo framing if it contributes to a frame.
Your output should be in a json format with the fields "frames-list", "reason" and "confidence", where "frames-list" is a list of frames that apply, "reason" is the reasoning for the frames chosen, and "confidence" is the confidence level (i.e. either low-confidence, medium-confidence, or high-confidence).
{"frames-list": "[<All frame names that apply from list provided above>], "reason": "<reasoning for the frames chosen>", "confidence": "<low-confidence/medium-confidence/high-confidence>"}
The frame names have to be ones provided in the list.
If there are no frames from the list that are clearly communicated, choose "None". If you choose "None", you do not choose any other frame.Output only the json and no other text.
"""


PROMPT_DICT_PIXTRAL = {
    'mini_multiple_frames': SYS_PROMPT + FRAMING_PROMPT + FRAMES_no_num + MINI_MULTIPLE_FRAMES_TASK_PROMPT,
    'actor_entity': SYS_PROMPT + prompt_entity
    # 'normal_multiple_frames': SYS_PROMPT + FRAMING_PROMPT + FRAMES_no_num + MULTIPLE_FRAMES_TASK_PROMPT
}