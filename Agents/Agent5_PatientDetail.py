from Utils.API_utils import gpt4_vision_caption, gemini_vision_caption

def Patient_Detail_Agent(question, image_path, RetrievedContent):
   
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
    
                Patient Detail: asks about the illness, status, age, gender, or any identity-related information of the patient.
                Chain 1: Deconstruct the question and identify key directives by first breaking down the question to identify specific patient-related details being asked (e.g., age, gender, illness, status). Highlight keywords that guide what to search for in the image (e.g., "What is the patient's status?" or "Identify the age and gender").
                Chain 2: Analyze the image for critical evidence relevant to the question (e.g., patient monitors, labels, metadata, surgical annotations). Ignore irrelevant details (e.g., background elements or unrelated surgical tools).
                Chain 3: Match the extracted visual details (e.g., demographic data, vitals) with the textual cues and answer choices and weigh each piece of evidence by its relevance to the question and its clarity in supporting or eliminating answer choices.
                Chain 4: Rank the answer choices based on how well they align with the prioritized visual evidence and retrieved medical knowledge. Interpret any unclear or partial cues using domain knowledge to refine how strongly each option is supported.
                Chain 5: Revisit the critical evidence and remove answer choices that conflict with the visual findings, procedural reasoning, or retrieved medical knowledge. Ensure the leading option remains logically and medically consistent.
                Chain 6: Confirm that the selected option is the most strongly supported. Justify the selection by linking it explicitly to the image, retrieved medical knowledge and question directives.
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
