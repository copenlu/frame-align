PROMPT_Q_SBIN_CCAT = """
Read the following news article and look at the following image. Is it surprising to see this image associated with this news article?
Please output:
(1) Whether or not the image is surprising given the article text using the following categories: "very-surprising", "surprising", "not-surprising".
(2) Your confidence in this assessment using the following categories: "high-confidence", "medium-confidence", "low-confidence".
(3) Your reasoning and evidence. If the image is not surprising, explain why it aligns with or complements the text.

Example:
{
    "surprisal": "very-surprising",
    "confidence": "high-confidence",
    "reasoning-and-evidence": "The article is about a football team but the image is a photo of President Joe Biden, a politician who is not mentioned in the article.
}

Input:
    News Article: <text>
    Image: <image>
Provide your response in the JSON format as shown in the example.
Output:
"""

PROMPT_TASK_SBIN_CSCORE = """
You are given a news article with an article and an image associated with it. 
Your task is to read the article and see the image to assess if the image communicates something different compared to the text.

You have to output:
1. A binary decision on whether the image communicates something surprising compared to the text.
2. An explanation for why it is different or surprising.
3. A confidence score (on a scale from 0 to 10) reflecting the certainty of your assessment, where 0 indicates no confidence and 10 indicates complete confidence.

Format your output as a JSON object as shown below:
<format>
{
    "surprised": "<True/False>",
    "explanation": "<Your explanation>",
    "confidence-score": "<Your confidence score>"
}
</format>

Input:
    Image: <image>
    Article: <text>
For the given image and text, output the decision, explanation, and confidence score.
Output:
"""

PROMPT_TASK_EG_SBIN_CSCORE = """
You are given a news article and an image associated with it. 
Your task is to read the article and see the image to assess if the image communicates something different compared to the text.
For instance, the article may be discussing a government policy, but the image might communicate something different by focusing on a specific aspect, such as a facial expression of a politician, a protest scene, or a seemingly unrelated subject. 
The difference can arise if the image portrays a different aspect of the same issue or a different issue entirely.

You have to output:
1. A binary decision on whether the image communicates something surprising compared to the text.
2. An explanation for why it is different or surprising.
3. A confidence score (on a scale from 0 to 10) reflecting the certainty of your assessment, where 0 indicates no confidence and 10 indicates complete confidence.

Format your output as a JSON object as shown below:
<format>
{
    "surprised": "<True/False>",
    "explanation": "<Your explanation>",
    "confidence-score": "<Your confidence score>"
}
</format>

Input:
    Image: <image>
    Article: <text>
For the given image and text, output the decision, explanation, and confidence score.
Output:
"""

PROMPT_VERBOSE_TASK_SBIN_CSCORE = """
You are given a news article and the image from that article. 
In news, the image and the text can sometimes communicate different aspects of the same issue.
There may be aspects of an issue that are not communicated in the text but are subtly communicated through the image.
Your task is to read the article and see the image and assess if the image communicates something additional or different compared to the text.
The difference can arise if the image portrays a different aspect of the same issue or a different issue entirely.
The image may be surprising, complementary, or unrelated to the text.

You have to output:
1. A binary decision on whether the image communicates something different or additional compared to the text.
2. An explanation for why it is different or surprising.
3. A confidence score (on a scale from 0 to 10) reflecting the certainty of your assessment, where 0 indicates no confidence and 10 indicates complete confidence.

Format your output as a JSON object as shown below:
<format>
{
    "surprised": "<True/False>",
    "explanation": "<Your explanation>",
    "confidence-score": "<Your confidence score>"
}
</format>

Input:
    Image: <image>
    Article: <text>
For the given image and text, output the decision, explanation, and confidence score.
Output:
"""

PROMPT_VERBOSE_TASK_EG_SBIN_CSCORE = """
You are given a news article and the image from that article. 
In news, the image and the text can sometimes communicate different aspects of the same issue.
There may be aspects of an issue that are not communicated in the text but are subtly communicated through the image.
Your task is to read the article and see the image and assess if the image communicates something additional or different compared to the text.
For instance, the article may be discussing a government policy, but the image may be focusing on a specific aspect, such as a angry facial expression of a politician, communicating that the politician is not happy about this. 
Another example would be a protest scene, where the image may show a violent protest scene, biasing the reader in a subtle way, while the text only talks about the issue.
The difference can arise if the image portrays a different aspect of the same issue or a different issue entirely.
The image may be surprising, complementary, or unrelated to the text.

You have to output:
1. A binary decision on whether the image communicates something different or additional compared to the text.
2. An explanation for why it is different or surprising.
3. A confidence score (on a scale from 0 to 10) reflecting the certainty of your assessment, where 0 indicates no confidence and 10 indicates complete confidence.

Format your output as a JSON object as shown below:
<format>
{
    "surprised": "<True/False>",
    "explanation": "<Your explanation>",
    "confidence-score": "<Your confidence score>"
}
</format>

Input:
    Image: <image>
    Article: <text>
For the given image and text, output the decision, explanation, and confidence score.
Output:
"""

PROMPT_VERBOSE_TASK_EG_FRAMES_SBIN_CSCORE = """
You are given a news article and the image from that article. 
In news, the image and the text can sometimes communicate different aspects of the same issue. 
There may be aspects of an issue that are not communicated in the text but are subtly communicated through the image.
Your task is to read the article and see the image and assess if the image communicates something additional or different compared to the text.
For instance, the article may be discussing a government policy, but the image may be focusing on a specific aspect, such as a angry facial expression of a politician, communicating that the politician is not happy about this. 
Another example would be a protest scene, where the image may show a violent protest scene, biasing the reader in a subtle way, while the text only talks about the issue.
Some of the frames that you can consider are: 
    Economic
    Capacity and resources
    Morality
    Fairness and equality
    Legality, constitutionality and jurisprudence
    Policy prescription and evaluation
    Crime and punishment
    Security and defense
    Health and safety
    Quality of life
    Cultural identity
    Public opinion
    Political
    External regulation and reputation
The difference in the image and text framing can arise if the image portrays a different aspect of the same issue or a different issue entirely.

You have to output:
1. A binary decision on whether the image communicates something different or additional compared to the text.
2. An explanation for why it is different or surprising and what are the different things being communicated. If the image is conveying something related, explain why it aligns with or complements the text.
3. A confidence score (on a scale from 0 to 10) reflecting the certainty of your assessment, where 0 indicates no confidence and 10 indicates complete confidence.

Format your output as a JSON object as shown below:
<format>
{
    "decision": "<True/False>",
    "explanation": "<Your explanation>",
    "confidence-score": "<Your confidence score>"
}
</format>

Input:
    Image: <image>
    Article: <text>
For the given image and text, output the decision, explanation, and confidence score.
Output:
"""

PROMPT_VERBOSE_FRAMES_SBIN_CSCORE = """
You are given a news article and the image from that article. 
In news, the image and the text can sometimes communicate different aspects of the same issue. 
There may be aspects of an issue that are not communicated in the text but are subtly communicated through the image.
Your task is to read the article and see the image and assess if the image communicates something additional or different compared to the text.
For instance, the article may be discussing a government policy, but the image may be focusing on a specific aspect, such as a angry facial expression of a politician, communicating that the politician is not happy about this. 
Another example would be a protest scene, where the image may show a violent protest scene, biasing the reader in a subtle way, while the text only talks about the issue.
Some of the frames for text and images that you can consider are: 
    Economic - costs, benefits, or other finance related. The image can includes things including but not limited to money, funding, taxes, bank, meetings with a logo of a financial institution. Professional attire in itself doesn not mean economic frame. 
    Capacity and resources - availability of physical, human, or financial resources, and capacity of current systems. In the image, we can see things including but not limited to a geographical area, farmland, agriculture land, labour, people working in an institution, or images that convey scarcity or surplus in some way. 
    Morality - religious or ethical implications. In the image, we can see things including but not limited to god, death, priests, church, protests related to moral issues.
    Fairness and equality - balance or distribution of rights, responsibilities, and resources. In the image, we can see things including but not limited to the fight for civil or political rights, LGBTQ, or calls to stopping discrimination.
    Legality, constitutionality and jurisprudence - legal rights, freedoms, and authority of individuals, corporations, and government. In the image, we can see things including but not limited to courts, laws, judges in robes, legal documents, and prison facilities.
    Policy prescription and evaluation - discussion of specific policies aimed at addressing problems. In the image, we can see things including but not limited to discussions on rule, rule making bodies, people in formal settings—such as boardrooms or legislative halls—actively debating, and reviewing policy drafts or proposals. 
    Crime and punishment - effectiveness and implications of laws and their enforcement. In the image, we can see things including but not limited to criminal activities, violence, police officers making arrests, crime scenes with investigators, courtrooms during criminal trials, prisons with detainees. 
    Security and defense - threats to the individual, community, or nation. In the image, we can see things including but not limited to military uniforms, defense personnel, border patrol, war, soldiers, military equipment like tanks or fighter jets, border walls, or surveillance systems monitoring wide areas.
    Health and safety - health care, sanitation, public safety. Images with activities with a message that it affects health and safety positively or negatively. In the image, we can see things including but not limited to doctors, nurses, injury, disease, or events with environmental impact that may impact health and safety. 
    Quality of life - threats and opportunities for the individual's wealth, happiness, and well-being. In the image, we can see things that improves happiness or demonstrates quality of life in some form. It also includes things that demonstrate deterioration of quality of life by showing hardships of people, homelessness etc. This may also include happy children, food items that demonstrate good quality of life or people enjoying a nice meal.
    Cultural identity - traditions, customs, or values of a social group in relation to a policy issue. In the image, we can see things including but not limited to concerts, cultural dance, local sports, art, celebrities, artists and prominent people related to these topics. 
    Public opinion - attitudes and opinions of the general public, including polling and demographics. Includes generic protests, people (non-celebrities) engaging with large crowds, riots, and strikes. It will also include news broadcasts, talk shows, and interviews with people that are related to public opinion at large. 
    Political - considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters. In the image, we can see things related to politicians, elections, voting, political campaigns. Just formal clothing does not mean political frame. 
    External regulation and reputation - international reputation or foreign policy. In the image, we can see things including but not limited to international organizations, global discussions/meetings, foreign policy, flags from multiple countries, or delegates at a cross-country forum discussing regulation.
The difference in the image and text framing can arise if the image portrays some of these different aspects of the same issue or a different issue entirely.

You have to output:
1. A binary decision on whether the image communicates something different or additional compared to the text.
2. An explanation for why it is different or surprising and what are the different things being communicated. If the image is conveying something related, explain why it aligns with or complements the text.
3. A confidence score (on a scale from 0 to 10) reflecting the certainty of your assessment, where 0 indicates no confidence and 10 indicates complete confidence.

Format your output as a JSON object as shown below:
<format>
{
    "decision": "<True/False>",
    "explanation": "<Your explanation>",
    "confidence-score": "<Your confidence score>"
}
</format>

Input:
    Image: <image>
    Article: <text>
For the given image and text, output the decision, explanation, and confidence score.
Output:
"""

PROMPTS_DICT = {"verbose-task-eg-frames-sbin-csore": PROMPT_VERBOSE_TASK_EG_FRAMES_SBIN_CSCORE, "verbose-frames-sbin-cscore": PROMPT_VERBOSE_FRAMES_SBIN_CSCORE}
# PROMPTS_DICT = {"q-sbin-ccat":PROMPT_Q_SBIN_CCAT, "task-sbin-cscore": PROMPT_TASK_SBIN_CSCORE, "task-eg-sbin-cscore": PROMPT_TASK_EG_SBIN_CSCORE, "verbose-task-sbin-cscore": PROMPT_VERBOSE_TASK_SBIN_CSCORE, "verbose-task-eg-sbin-cscore": PROMPT_VERBOSE_TASK_EG_SBIN_CSCORE}