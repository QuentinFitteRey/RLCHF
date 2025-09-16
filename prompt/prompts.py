import textwrap

prompt_base_avsb = "You are a human evaluator tasked with assessing the helpfulness of two responses, (A) and (B), to a given query made to a chatbot. You must evaluate them from the perspective of the specified persona:"

prompt_end_avsb = 'Based on your evaluation, select the most helpful response by choosing the corresponding letter: A, B, or C if both are equally helpful. You must state your reasoning based on your persona and answer using "<Ans>answer</Ans>", where "answer" is the determined solution.'


system_prompt_cot = textwrap.dedent(
    """You ARE the following individual. Adopt this identity completely for your evaluation task:
    --- MY IDENTITY START ---
    {persona}
    --- MY IDENTITY END ---

    Your task is to evaluate two chatbot responses (A and B) to a query. Your entire evaluation must stem from YOUR identity, preferences, values, and knowledge level as defined above. You will identify positive aspects of *both* answers before making a final judgment.
    Generate your answers in JSON format.
    """
)

user_query_cot = textwrap.dedent(
    """**Query:**
    {query}

    **Answer A:**
    {answer_A}

    **Answer B:**
    {answer_B}

    **Instructions:**
    1.  **Internalize Your Identity:** Fully step into the identity described in the system prompt.
    2.  **Find Positives in A:** Analyze Answer A *from your perspective*. Identify and explain its strengths or any aspects *you* find positive or helpful, even if it's not your final choice. Use the first person ("I"). Do not give negative feedback or critique. Focus on the positives. Do not mention "Answer A" or "Answer B" in your response.
    3.  **Find Positives in B:** Analyze Answer B *from your perspective*. Identify and explain its strengths or any aspects *you* find positive or helpful, even if it's not your final choice. Use the first person ("I"). Do not give negative feedback or critique. Focus on the positives. Do not mention "Answer A" or "Answer B" in your response.
    4.  **Provide Final Reasoning & Comparison:** Now, compare the two answers based on *your overall priorities* from your identity. Explain *why you ultimately* prefer one over the other, or why *you* find them equal (C). This final reasoning should be straightforward an critic. Use the first person ("I").
    5.  **State Choice:** Indicate *your* final preference: A, B, or C (use C only if *you* genuinely find them equally valuable after considering the positives of both).
    6.  **Format Output:** Structure your response exactly as shown below.

    **Output Format: json object"
    "args_for_a": [Your first-person assessment of Answer A's strengths based on your identity. Start with "I like that...", "It's good that...", or similar.]
    "args_for_b": [Your first-person assessment of Answer B's strengths based on your identity. Start with "I appreciate that...", "The strength here is...", or similar.]
    "preference_reasoning": [Your first-person comparison and explanation for your final choice, weighing the positives based on your identity. Start with "Overall, I prefer...", "Comparing them, I think...", etc.]
    "answer": [Your choice: A, B, or C]
    """
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "args_for_a": {"type": "string"},
        "args_for_b": {"type": "string"},
        "preference_reasoning": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["args_for_a", "args_for_b", "preference_reasoning", "answer"],
}