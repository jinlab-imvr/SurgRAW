import re
import sys
import logging
from Utils.API_utils import call_gpt35Turbo_api
from Agents.Agent1_ActionRecognition import Action_Recognition_Agent
from Agents.Agent2_InstrumentIdentification import Instrument_Recognition_Agent
from Agents.Agent3_ActionPrediction import Action_Prediction_Agent
from Agents.Agent4_SurgicalOutcome import Surgical_Outcome_Agent
from Agents.Agent5_PatientDetail import Patient_Detail_Agent
from Agents.RAG_module import query_rag
from Agents.GP_Moderator import multi_agent_debate

def classify_overall_question(question):
    """
    Classify the question as 'vision-based' or 'knowledge-based' using GPT-3.5
    """
    prompt = f"""
    You are an expert surgical question classifier. Classify the following question as either "vision-based" or "knowledge-based".
    Vision-based questions relate to tasks that require analyzing visual aspects (e.g., instrument identification or action recognition or ongoing action).
    Knowledge-based questions relate to tasks that require external clinical or procedural context (e.g., surgical plan, outcome, or patient details).

    Question: "{question}"

    Answer with exactly one word: either "vision-based" or "knowledge-based".
    """
    result = call_gpt35Turbo_api(prompt).strip().lower()
    return result

def classify_vision_question(question):
    """
    If vision-based, classify as:
      'instrument identification' or 'action recognition'
    """
    prompt = f"""
    You are an expert surgical question classifier. Given that the question is vision-based, determine if it is asking about:
    (a) "instrument recognition"  - identifying the surgical tool(s) visible in the image.
    (b) "action recognition"         - understanding what surgical action is being performed in the image.

    Question: "{question}"

    Answer with exactly one of the following terms (all lower case): "instrument recognition" or "action recognition".
    """
    result = call_gpt35Turbo_api(prompt).strip().lower()
    return result

def classify_knowledge_question(question):
    """
    If knowledge-based, classify as:
      'surgical plan', 'outcome', or 'patient detail'
    """
    prompt = f"""
    You are an expert surgical question classifier. Given that the question is knowledge-based, determine which of the following categories it belongs to:
    (a) "action prediction"      - questions about what the surgeon plans to do next.
    (b) "outcome"            - questions about the expected surgical result.
    (c) "patient detail"     - questions about patient characteristics or demographics.

    Question: "{question}"

    Answer with exactly one of the following terms (all lower case): "action prediction", "outcome", or "patient detail".
    """
    result = call_gpt35Turbo_api(prompt).strip().lower()
    return result

def final_orchestrator(question, image_path):
    """
    Collect each step in a list of conversation steps.
    Returns:
      {
        "steps": List[ (role: str, text: str), ... ],
        "final_result": str or dict
      }
    """
    steps = []

    # 1) Department Coordinator
    steps.append(("dept_coordinator",
                  f"Department Coordinator: Received question: '{question}'."))

    overall_class = classify_overall_question(question)
    steps.append(("dept_coordinator",
                  f"Classified task as: **{overall_class}**."))

    if overall_class not in ["vision-based", "knowledge-based"]:
        # fallback
        # steps.append(("dept_coordinator","[ERROR] Unrecognized classification. Defaulting to 'vision-based'."))
        overall_class = "vision-based"

    agent_function = None
    retrieved_content = None

    # 2) Department Heads
    if overall_class == "vision-based":
        # Vision Dept Head
        vision_class = classify_vision_question(question)
        steps.append(("vision_dept_head", f"Vision Dept Head: question → **{vision_class}**."))

        if vision_class == "instrument recognition":
            steps.append(("vision_dept_head", "→ Routing to Instrument Specialist."))
            agent_function = Instrument_Recognition_Agent
        elif vision_class == "action recognition":
            steps.append(("vision_dept_head", "→ Routing to Multi-Agents Panel Discussion (Action Interpreter)."))
            agent_function = multi_agent_debate
        else:
            # steps.append(("vision_dept_head", f"[ERROR] Unrecognized: {vision_class}, fallback to action recognition."))
            agent_function = multi_agent_debate

    else:
        # Knowledge Dept Head
        knowledge_class = classify_knowledge_question(question)
        steps.append(("knowledge_dept_head", f"Knowledge Dept Head: question → **{knowledge_class}**."))

        if knowledge_class == "action prediction":
            steps.append(("knowledge_dept_head", "→ Routing to Action Predictor."))
            agent_function = Action_Prediction_Agent
        elif knowledge_class == "outcome":
            steps.append(("knowledge_dept_head", "→ Routing to Outcome Analyst."))
            agent_function = Surgical_Outcome_Agent
        elif knowledge_class == "patient detail":
            steps.append(("knowledge_dept_head", "→ Routing to Patient Advocate."))
            agent_function = Patient_Detail_Agent
        else:
            # steps.append(("knowledge_dept_head", f"[ERROR] Unrecognized: {knowledge_class}, fallback to SurgicalPlan_Agent."))
            agent_function = Action_Prediction_Agent

        # Query RAG
        steps.append(("knowledge_dept_head", "[INFO] Querying RAG for external knowledge..."))
        retrieved_content = query_rag(question)
        snippet = retrieved_content[:300] + "..." if retrieved_content else "No data."
        steps.append(("knowledge_dept_head", f"RAG snippet:\n{snippet}"))

    # 3) Execute Agent
    steps.append(("agent", f"Executing **{agent_function.__name__}**..."))

    if overall_class == "knowledge-based":
        final_answer = agent_function(question, image_path, retrieved_content)
    else:
        final_answer = agent_function(question, image_path)

    # 4) Add the agent's final answer as a step
    if isinstance(final_answer, dict):
        # e.g. multi_agent_debate might produce:
        # { "instrument_agent_answer": "...", "action_agent_answer": "...", "metrics": {...} }
        # Let's add them as separate conversation steps
        instr_ans = final_answer.get("instrument_agent_answer", "No instrument answer.")
        act_ans   = final_answer.get("action_agent_answer", "No action answer.")
        metrics   = final_answer.get("metrics", {})
        steps.append(("agent_instrument_specialist",
                      f"Instrument Specialist:\n{instr_ans}"))
        steps.append(("agent_action_interpreter",
                      f"Action Interpreter:\n{act_ans}"))
        steps.append(("action_evaluator",
                      f"Evaluation Metrics:\n{metrics}"))
    else:
        # Single-agent final
        steps.append(("agent", f"Final Answer:\n{final_answer}"))

    # Return the entire conversation flow
    return {
        "steps": steps,
        "final_result": final_answer
    }
