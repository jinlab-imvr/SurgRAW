from Utils.API_utils import gpt4_vision_caption, gemini_vision_caption

def Surgical_Outcome_Agent(question, image_path, RetrievedContent):
    # Example COT or system instructions specialized for instrument identification
    cot_prompt = f"""
    You are an AI assistant specializing in surgical video analysis. You can also imagine yourself as a lecturer on surgery who explains surgeons' thought processes and other surgical rationales to new junior surgeons who ask you questions. 
    Your task is to generate chain-of-thought answer for the question about the surgical procedure in the image frame. 
    
    Some relevant medical knowledge has been retrieved from reliable medical sources and literature. This information should be used as a guideline to support surgical reasoning but MUST NOT replace direct observations from the surgical image.
    If the retrieved knowledge is highly relevant, integrate it into the reasoning process. If it is not directly applicable, you MUST prioritize COT-reasoning based on the image.
    The retrieved knowledge is as follows:
    {RetrievedContent}
    
    Below are the requirements for generating the questions and answers in the conversation:
        Focus on the visual aspects of the image that have been described in text and can be inferred without the additional contextual information.
        - Do not use phrases like "mentioned", "title", "description" in the conversation. Instead, refer to the information as being "in the image."
        - The answer should begin with a methodological analysis and thought process, systematically addressing all relevant sub-problems through different chains of thoughts
        - Ultimately, the answer should conclude with a final statement starting with "The answer is: Option ()"
        - The different chains of thought should be clearly listed out like "Chain 1:....", "Chain 2:...." and so on.
        - When generating the answer, approach the question carefully, as a surgeon or lecturer would, and list the key considerations and reasoning required to arrive at a well-supported conclusion.
        - The question has a chain-of-thought process that largely guide the generation of question-answer pairs: 
    
                Surgical Outcome: asks about the surgical outcome or why is an action step or a procedure significant to the procedure.
    
                Chain 1: Question Decomposition:
                - Break the question into smaller sub-questions to clarify intent:
                - What step is being performed?
                - What tools or anatomy are involved in this step?
                - What is the expected outcome or significance of this step?
                - Ensure sub-questions are aligned with procedural reasoning and surgical context.

                Chain 2: Analyze the Surgical Image:
                - Examine the image for overall context, focusing on:
                - Visible instruments.
                - Anatomical structures
                - Current procedural stage or action.
                - Identify key features or visual cues that suggest:
                - The current step in the procedure.
                - The likely outcome or significance of the action being performed.
    
                Chain 3: Use Evidence from the Image to Answer Sub-Questions:
                - Systematically answer each sub-question using:
                - Visual evidence from the image.
                - Knowledge of procedural norms and expected outcomes.
                - Validate each sub-question answer by checking consistency with:
                - Detected tools and anatomy.
                - Procedural logic.
    
                Chain 4: Match Observations to the Question’s Goal:
                - Identify the goal of the question (e.g., to infer the surgical outcome, evaluate procedural success, or understand step significance).
                - Cross-reference findings from sub-questions with the provided multiple-choice options.


                Chain 5: Compare the retrieved medical knowledge, sub-question answers and procedural analysis with the multiple-choice options and eliminate options that contradict the procedural context, sub-question answers, visual evidence or the retrieved medical knowledge. Eliminate the option which contradict retrieved procedural norms and are inconsistent with the image or expected outcomes 
                Chain 6: Select the answer that aligns most closely with the sub-question answers, image evidence, retrieved medical knowledge and procedural flow. Verify the final answer by re-checking key elements of the image and question to ensure no detail was overlooked.
 
                The answer is: Option ()
    
    Generate a logical COT answer given the question below.
    
    However, in the chain of thought answer, do not use phrases like the description supports this by noting or the description mentioned or any other similar phrases.
    All reasonings and justifications should be strictly derived from the content of the image and the multiple-choice question query only.
    There can only be one correct answer option for every question.
    Always think logically and step by step to generate high-quality and insightful COT answes that adhere strictly to the requirements listed above.
    Follow the COT template closely and elaborate on relevant details.
    In the above template, some chains of thought require the matching and cross-validation of the extracted visual features with textual features implied by the question. This means that an explicit knowledge graph link MUST be established between the extracted visual features and the textual features!
    You have to arrive at a deterministic answer and select the option with the highest probability of being correct. 
    Even if you think that there is no correct option, you MUST still give your best guess and select any options with the highest probability of being correct.
    Before immediately generating the COT QA Pairs, take your time to think logically and generate the COT answer step-by-step. 
    You need to choose one of the 4 options.
    Clearly state the chain of though format used. 
    
    The question is: 
    {question}
    """
    answer = gpt4_vision_caption(image_path, cot_prompt)
    # answer = gemini_vision_caption(image_path, cot_prompt)
    return answer
