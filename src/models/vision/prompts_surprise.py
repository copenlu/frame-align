PROMPT_OPENQ_CAT = f"""
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

PROMPT_TASK_SBIN_CSCORE = f"""
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

PROMPT_TASK_EG_SBIN_CSCORE = f"""
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

PROMPTS_DICT = {"openq-cat": PROMPT_OPENQ_CAT, "task-sbin-cscore": PROMPT_TASK_SBIN_CSCORE, "task-eg-sbin-cscore": PROMPT_TASK_EG_SBIN_CSCORE}