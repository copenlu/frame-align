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
    1: Economic - The costs, benefits, or any monetary/financial implication of the issue (to an individual, family, organization, community or to the economy as a whole) The economic frame includes mentions of: Money, funding, or taxes Socio-economic class (working-class, middle-class, rich, poor). In the image, we can see things including but not limited to bank, money, meetings with a logo of a financial institution. 
 

    2: Capacity and resources - The availability/lack of time, physical, geographical, human, and financial resources.  This frame stresses finite resources, and specifically deals with limitations or availability of resources. Think of it in terms of there being “not enough” or “enough” of something. Focus on availability/scarcity/surplus of physical resources (say in a farmland, classroom space, prisons, etc.), human resources (law enforcement, engineers, doctors, etc.) or financial resources (availability of credit, commercial items, funds, etc.) 
    
    In the image, we can see things including but not limited to a geographical area, labour, people working in an institution, or images that convey scarcity or surplus in some way.

 
    3: Morality -  Any perspective compelled by religious doctrine or interpretation, duty, honor, righteousness or any other sense of ethics or social responsibility (religious or secular). Morality frame refers to social responsibility (while fairness refers to social justice), Non-religious morality frames could include general moral imperatives to help others, Appeals that something “is just the right thing to do” or “would indicate a recognition of our shared humanity”, or arguing against something with “I don’t think it’s right”. The morality frame includes mentions of: a) Morals or ethics b) Humanitarian crisis c) Human rights d) Anything religious or related to religion. 
    
    In the image, we can see things including but not limited to god, death, protests related to humanitarian issues.

    4: Fairness and equality - The fairness, equality, or inequality with which laws, punishment, rewards, and resources are applied or distributed among individuals or groups.Fairness and Equality frame cues often focus on whether society and its laws are equally distributed and enforced across region, race, gender, economic class, etc.  Often used in discussing social justice issues. Fairness frame refers to social justice, while (morality refers to social responsibility). The fairness frame includes discussions of: a) Discrimination, racism, xenophobia b) Portrays certain social groups in a negative light c) Stereotypes or microaggressions 

    In the image, we can see things including but not limited to the fight for LGBTQ, reproductive rights, call for stopping discrimination

    
    5: Legality, constitutionality and jurisprudence - The legal, constitutional, or jurisdictional aspects of an issue. This includes: 1) Legal: court cases and existing laws that regulate policies 2) Constitutional: discussion of constitutional interpretation or potential revisions 3) Jurisdiction: which government body should be in charge of a policy decision and/or the appropriate scope of a body’s policy reach. 4) Legality and constitutionality of “rules” (federal law, business regulations, etc.) and court cases on whether the rules were upheld  5) All aspects of jurisdiction: US vs United Nations, state vs federal, voters vs courts, etc. 6) Proposed laws framed as jurisdictional issues from the outside (can states regulate immigration?) gets both policy and legality. 7) The legality frame does NOT include legislation, which comes under policy

    In the image, we can see things including but not limited to judges, prisons, laws, discussions on policy.
    
    
    6: Policy prescription and evaluation - Existing policies, policies proposed for addressing an identified problem, as well as analysis of whether hypothetical policies will work or existing policies are effective.
    Refers to “rules” (federal/state law, business policy, regulation, Congressional bills)
    The policy frame includes mentions or depictions of: 
    Deportation policy (also is crime & punishment)
    Brexit
    Trade deals are tagged with economic, external, and policy
    Proposals/actions to build border wall (both policy and security)
    Executive orders, declaration of state/national emergencies
    Tax reform (both economic and policy frames) 
    In the image, we can see things including but not limited to discussions on rule, rule making bodies, tax forms.

    
    7: Crime and punishment - The violation of policies in practice and the consequences of these violations. These include:
    Breaking the rules and/or getting punished 
    Crimes that cause physical harm are also tagged as health & safety
    Examples talking about:
    Gun violence (also health and safety) 
    Potential criminals facing prosecution (also legality)
    Sentences delivered in a court case (also legality)
    Punishments for law-breaking (fines are also tagged as economic)

    In the image, we can see things including but not limited to criminal activities, violence, and police presence.

    8: Security and defense - Any threat to a person, group, or nation. These include:
    Any defense that needs to be taken to avoid that threat, including tools and technologies
    Includes issues of national security including resource security, and efforts of individuals to secure homes, neighborhoods or schools 
    The security & defense frame includes mentions of: 
    Borders 
    Terrorism or risk of immigrants being terrorists 
    Invasions or descriptions of immigrants as invaders 
    Taking over a country (similar to invasion) 
        
    In the image, we can see things including but not limited to military uniforms, defense personnel, border patrol, war.

    9: Health and safety - Healthcare, Sanitation, Public safety. Includes:
    Health care access and effectiveness, illness, disease, sanitation, obesity, mental health, infrastructure/building safety
    Policies taken to ensure safety should a tragedy occur (emergency preparedness kits, lock down training, disaster awareness classes, etc)
    The health and safety frame also includes: 
    Disaster relief
    Medicine, vaccines, etc
    Medical and health organizations (e.g. CDC, NHS) 
    Physical harm: words like carnage, death, bodily wound, injury, bloodshed etc.
    Gun violence, killing, shooting, other violent crime also get crime frame
    Female genital mutilation (FGM) (also cultural identity)
    In the image, we can see things including but not limited to doctors, nurses, injury, disease.

    10: Quality of life - The benefits and costs of any policy on quality of life. Includes:
    The effects of a policy on people’s wealth (also economic), mobility, access to resources, happiness, social structures, ease of day-to-day routines, quality of community life, etc.
    Could include discussions of working conditions and terrible wages 
    Facing a “hostile environment”
    Poverty, homelessness, needing food stamps
    Day-to-day fears, job loss, being stopped by police, or not being able to participate in normal social activities due to these fears 
    Taking actions (such as working long hours) to improve their family’s quality of life (e.g. giving children educational opportunities)
    Generic statements about seeking a better life, looking for an escape, etc.
    Hardships in crossing the border or arriving in the United States get Health and Safety, NOT Quality of Life. 

    In the image, we can see things including but not limited to happiness, joy, hardships of people, homelessness.

    11: Cultural identity - Social norms, trends, values and customs constituting any culture as they relate to a specific policy issue. Includes discussions about or depictions of:
    Population changes, including replacement migration
    A nation’s values e.g. patriotism
    Personal (or familial) immigration stories and experiences
    Artwork (books, songs, etc.)
    Music/entertainment/foods
    Including participating in sports (as form of entertainment)
    Cultural norms or stereotypes of ethnic and political groups
    Trends, attitudes, or beliefs sweeping the nation 
    Associations with notable people in order to make a cultural reference
    Celebrity endorsements for policy issues 
    In the image, we can see things including but not limited to concerts, cultural dance, art, and prominent people related to these topics. 

    12: Public opinion - The public’s opinion
    Includes references to general social attitudes, protests, polling, as well as implied or actual consequences of diverging from or “getting ahead of” public opinion or polls.
    Includes references to a party’s base or constituency (would overlap with Politics) ○ Brexiteers, Remainers, Leavers, Trump supporters etc. when giving generalizations about how they feel and their opinions 
    Includes any public passage of a proposition/law
    Explicit mentions of referendum
    Includes protests, riots, and strikes (incl. hunger strikes)
    Sharing petitions and encouraging people to take political action 
    
    
    13: Political - Any political considerations surrounding an issue 
    Includes issue actions or efforts or stances that are political, such as partisan filibusters, lobbyist involvement, bipartisan efforts, deal-making and vote-trading, appealing to one’s base, explicit statements that a policy issue is good or bad for a particular political party
    Discussions of political maneuvering, partisan conflicts
    Mentions or depictions of a political entity or political party:
    Political entities, parties, partisan conflict (see above)
    Voting and elections
    Political debates
    Lobbying or campaigning 
    Socialism, fascism, etc. when discussed as political philosophies 
    In the image, we can see things including but not limited to politicians, elections, voting, political campaigns. 

    14: External regulation and reputation - A nation’s external relations with another nation. The external frame only focuses on explicit relationships between countries and states, not just any mention of something happening in another country. For instance:
    International efforts to achieve policy goals, alliances or disputes between groups
    Anything about the United Nations (UN) and its organizations (such as World Health Organization)
    Anything about globalization, globalism, globalists, (even if you suspect it’s just there as an antisemitic dogwhistle)
    In the image, we can see things including but not limited to international organizations, global discussions, cross country forums. 


    15: None - no frame could be identified because of lack of information in the image."""

    # 15: Other - frame is present and can be any coherent group of frames not covered by the above categories.

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

PROMPT_DICT_PIXTRAL = {k: prefix_instruction + v for k, v in PROMPT_DICT_PIXTRAL.items()}