"""Prompts for T2_Text (image + text question). prompt_id is used in preds_path()."""

GOLD_STYLE = """\
You are an expert on Middle Eastern and North African culture. Look at the image and answer the question about it.

Write the answer in this style:
- One or two declarative sentences, about 20-30 words in total. Never more than 45 words.
- Begin with "The <subject> ..." and restate the subject of the question (for example "The lanterns shown ...", "The dish in the image ...").
- Name the specific entity you recognise (landmark, dish, garment, airline, city, country) and add one short clause of cultural or historical context.
- If you are not fully sure, hedge mildly ("likely", "appears to be", "suggests") instead of refusing.
- Identification questions ("What is", "Which", "Name") get a short answer, around 20 words. Questions about significance, symbolism or meaning get a fuller answer, around 30 words.
- Plain text only: no yes/no opening, no "Answer:", no bullets, no markdown. End with a period.

Question: {question}"""

PROMPTS = {"plain": "{question}", "gold_style_v1": GOLD_STYLE}


def build_prompt(question: str, prompt_id: str = "gold_style_v1") -> str:
    return PROMPTS[prompt_id].format(question=question.strip())
