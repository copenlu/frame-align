prefix_instruction = """

USER: You are an intelligent and logical assistant. Your job is to see the image and then read the question. You need to answer the question based on the image. If the answer could not be answered using just the image, you should put it as "None.
"""

prompt_a = """

Caption the shown image in detail and do not forget to include the people, objects, and the scene in the image if any.

<format>
The format of the output should be as a json file that looks follows:
{
    "caption": "<caption>"
}
</format>

<examples>
{
    "caption": "A group of people are sitting on a bench in a park. There are trees in the background and a dog is running in the foreground."
}

{
    "caption": "The image shows a man in a suit standing in front of a large building called the White House. There are American flags on either side of the building."
}
</examples>

 And now caption the image in detail based on the image you see.


"""

prompt_b = """

Categorise the shown image in one of the following categories: 1) Scenery (2) Actors (3) Symbols . Also give a brief explanation of why you chose the category.

<format>
The format of the output should be as a json file that looks follows:
{
    "category": "<category>"
    "explanation of the category": "<explanation>"
}

</format>

<examples>
{   
    "category": "Scenery"
    "explanation of the category": "The image shows a beautiful landscape with mountains and a river in the background."
}

{
    "category": "Actors"
    "explanation of the category": "The image shows a group of people who are the main focus of the image. They look like they are having a conversation."
}

{
    "category": "Symbols"
    "explanation of the category": "The image shows a logo of united nations. It has a globe in the center and olive branches around it."
}

</examples>

 And now for the image you see, categorise the image into: a) Scenery b) Actors c) Symbols. Also give a brief explanation of why you chose the category.


"""

prompt_c = """

In the shown image, who are the actors (people of importance)? Actors are identifiable individuals, collectives, or institutions, usually mentioned by name, which are not only subject of the article but who are given the opportunity - via direct or indirect speech - to communicate their point of view. Actors can be persons, groups, committees, organizations, or institutions. Journalists can be coded as actors as well when they not merely act as chronicler of events and statements but add context, interpretation and/or evaluation to the article, indicated by a statement which is not solely based or does not merely sum up interpretations and/or evaluations by other quoted actors. 

<format>
The format of the output should be as a json file that looks follows:
{
    "actor": "<actor>"
}

</format>

<examples>
{
    "actor": "The President"
}

{
    "actor": "the African Union"
}

{
    "actor": "environmental NGO Thinktank"
}

</examples>

 And now for the actors in the image, who are the actors in the image. 


"""

prompt_d = """

Mention roles of the actors in the image play in the society. . Actors are identifiable individuals, collectives, or institutions, usually mentioned by name, which are not only subject of the article but who are given the opportunity - via direct or indirect speech - to communicate their point of view. Actors can be persons, groups, committees, organizations, or institutions. Journalists can be coded as actors as well when they not merely act as chronicler of events and statements but add context, interpretation and/or evaluation to the article, indicated by a statement which is not solely based or does not merely sum up interpretations and/or evaluations by other quoted actors. 

<format>
The format of the output should be as a json file that looks follows:
{
    "actor": "<actor>"
    "role": "<role>"
}

</format>

<examples>
{
    "actor": "The President"
    "role in society": "The President is the head of the country and is responsible for making important decisions that affect the citizens."
}

{
    "actor": "The environmental NGO Thinktank"
    "role in society": "The environmental NGO Thinktank is responsible for researching and advocating for policies that protect the environment."
}

</examples>

 And now for the actors in the image, what roles the actors in the image play in the society. 



"""

prompt_e = """

In the shown image, what symbolic meaning image is trying to convey. Symbolic meaning are hidden meaning in the images not explicitly shown in the images.

<format>
The format of the output should be as a json file that looks follows:
{
    "symbolic thing": "<symbolic thing>"
    "symbolic meaning": "<symbolic meaning>"
}

</format>

<examples>

{
    "symbolic thing in image": "The color red"
    "symbolic meaning": "The color red in the image symbolizes danger and passion."
}

{   
    "symbolic thing in image": "raised fist"
    "symbolic meaning": "The raised fist in the image symbolizes power and unity."
}

</examples>

 And now for the image you see, what symbolic meaning is the image trying to convey. 


"""

prompt_f = """

For each actor present in the image, are they represented a) positively b) neutral or c) negatively. Give a brief explanation of why you chose the category.

<format>
The format of the output should be as a json file that looks follows:
{
    "actor": "<actor>"
    "representation": "<representation>"
    "explanation": "<explanation>"
}

</format>

<examples>
{
    "actor": "The President"
    "representation": "Positive"
    "explanation": "The President is shown smiling and shaking hands with people. This suggests that they are friendly and approachable."
}

{
    "actor": "The CEO of a company"
    "representation": "Negative"
    "explanation": "The CEO is shown frowning and looking angry. This suggests that they are unhappy or upset about something."
}

</examples>

 And now for each actor present in the image, are they represented positively, neutral or negatively. Give a brief explanation of why you chose the category. 


"""

prompt_g = """

How many people in the image.

<format>
The format of the output should be as a json file that looks follows:
{
    "number of people": "<number of people>"
}
</format>

<examples>
{
    "number of people": "3"
}

{
    "number of people": "5"
}

</examples>

  And now for the image you see, how many people are there. 

"""

prompt_h ="""

What are the facial expressions of the people in the image? Describe each person and their facial expressions. Say 'None' if none exists.

<format>
The format of the output should be as a json file that looks follows:
{
    "actor 1": <description of the actor 1>
    "facial expression": "<facial expression>"

    "actor 2": <description of the actor 2>
    "facial expression": "<facial expression>"
}
</format>

<examples>
{
    "actor 1": "the man in the suit"
    "facial expression": "smiling"

    "actor 2": "the police officer"
    "facial expression": "serious"

}
</examples>

 Now see the image shown and describe people and their facial expressions. Say 'None' if none exists. 


"""

prompt_i ="""

What is the perceivable gender in the image? Perceivable gender is the way others view a person along a continuum from masculine to feminine. It is based on the person's appearance, behavior, and other characteristics.

<format>
The format of the output should be as a json file that looks

{
    "description of the person 1": "<description of the person 1>"
    "perceived gender 1": <perceived gender 1>
    

    "description of the person 2": "<description of the person 2>"
    "perceived gender 2": <perceived gender 2>
}

</format>

<examples>

{
    "description of the person 1": "the person in the suit"
    "perceived gender 1": "male"

    "description of the person 2": "a boy in a dress"
    "perceived gender 2": "queer"

    "description of the person 3": "a person with short hair"
    "perceived gender 3": "non-binary"

    "description of the person 4": "a woman with long hair in a dress"
    "perceived gender 4": "female"

}

</examples>

 And now for the image you see, what is the perceived gender in the image? [INST]


"""

prompt_j ="""

Are there actors in the image shown in a position of power? Example: Looking down vs looking up at someone, looking straight in the eye = honesty, looking away = suspect.

<format>
The format of the output should be as a json file that looks follows:
{
    "actor": "<actor>"
    "position of power": "<yes or no>"
    "explanation": "<explanation>"
}

</format>

<examples>
{
    "actor": "The President"
    "position of power": "Yes"
    "position of power": "Looking straight in the eye with seriousness"
}

{
    "actor": "The children in the image"
    "position of power": "No"
    "position of power": "Looking up at the teacher with curiosity"
}

</examples>

 And now for the actors in the image, are they shown in a position of power?


"""

prompt_k = """

What is the level of intimacy between the subjects in the image. The level of intimacy is determined by the distance between the subjects in the image.
a) extreme close up between the subjects = high
b) distance between the subjects = low
<format>
The format of the output should be as a json file that looks follows:
{
    "level of intimacy": "<level of intimacy>"
}

</format>

<examples>
{
    "level of intimacy": "low"
    "explanation": "The image shows a group of people standing far apart from each other. This suggests that there is a low level of intimacy with the subject."
}

{
    "level of intimacy": "high
    "explanation": "The actors in the image stand close to each other. This suggests that there is a high level of intimacy with the subject."
}

</examples>

 And now for the image you see, what is the level of intimacy between the subjects in the image.


"""

prompt_l = """ 
What emotions does the image convey? Emotions are the feelings that the image evokes in the viewer. Give a brief explanation of why you chose the emotion.

<format>
The format of the output should be as a json file that looks follows:
{
    "emotion": "<emotion>"
    "explanation": "<explanation>"
}

</format>

<examples>
{
    "emotion": "happiness"
    "explanation": "The image shows people smiling and laughing. This suggests that the image conveys happiness."
}

{
    "emotion": "sadness"
    "explanation": "The image shows a person crying. This suggests that the image conveys sadness."
}

</examples>

 And now for the image you see, what emotions does the image convey?


"""

prompt_m = """

What emotions are conveyed by the people in the image? Describe each person and the emotions they are conveying. Say 'None' if none exists.

<format>
The format of the output should be as a json file that looks follows:
{
    "actor 1": <description of the actor 1>
    "emotion": "<emotion>"
    "explanation": "<explanation>"

    "actor 2": <description of the actor 2>
    "emotion": "<emotion>"
    "explanation": "<explanation>"
}

</format>

<examples>
{
    "actor 1": "the politician standing in the image"
    "emotion": "serious"
    "explanation": "The politician looks serious and focused. This suggests that they are thinking deeply about something."

    "actor 2": "the child playing in the image"
    "emotion": "happy"
    "explanation": "The child looks happy and excited. This suggests that they are having fun."

}

</examples>

 And now for the image you see, describe each person and the emotions they are conveying. Say 'None' if none exists.

"""


# add prefix_instruction to all the prompts
PROMPT = [prompt_a, prompt_b, prompt_c, prompt_d, prompt_e, prompt_f, prompt_g, prompt_h, prompt_i, prompt_j, prompt_k, prompt_l, prompt_m]

PROMPT_LIST_GEMMA = [prefix_instruction + prompt for prompt in PROMPT]
