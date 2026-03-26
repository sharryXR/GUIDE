import base64
from langchain_openai import ChatOpenAI
import math
import random
from PIL import Image, ImageDraw
import re
import os
from pathlib import Path
import webvtt
import av
from io import BytesIO
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
# def construct_prompt(screenshot1_path, json_file_1, screenshot2_path, json_file_2, task_description, thought):
#     prompt = f"""
# You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. Your goal is to compare the screenshots, analyze changes in annotated interactive elements using both visual information and provided JSON descriptions, and infer the most likely actions based on the given context. The output should be a sequence of natural language descriptions of the actions or an indication that no action occurred.

# Inputs:

# Screenshot 1: {screenshot1_path}
# JSON File 1: {json_file_1}
# Screenshot 2: {screenshot2_path}
# JSON File 2: {json_file_2}
# Task Description: {task_description}
# Thought: {thought}

# Analysis Steps:
# 1. Compare Screenshots Visually: Examine the provided screenshots to identify visual changes, focusing on annotated interactive elements (e.g., buttons, text fields).
# 2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
# 3. Incorporate Context: Use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
# 4. Infer Action Sequence: Determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.
# 5. Output each action in natural language, e.g., "Clicked the 'Login' button", "Typed 'john_doe' into the 'Username' field", or "Scrolled down the page". If no significant change is detected, conclude there is "No action".

# Action Space:
# - click(start_box='<|box_start|>(x1,y1)<|box_end|>')
# - left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
# - right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
# - drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
# - hotkey(key='')
# - type(content='') #If you want to submit your input, use \"\\" at the end of `content`.
# - scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
# - wait() #Sleep for 5s and take a screenshot to check for any changes.
# - finished()
# - call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

# Output Format:
# Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
# Actions: [A sequence of action types, e.g., ["Click", "Type"], or "No action"]
# Action NLP Description: [Natural language description of the actions, e.g., "Clicked the 'Submit' button", "Typed 'john_doe' into the 'Username' field", or "No action"]

# Guidelines:
# - Use clear and user-friendly natural language.
# - If no relevant changes or task-related changes occur, output "No action".
# - If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
# - Prioritize primary task-relevant actions when multiple changes are present.
# - Rely explicitly on provided JSON data for accuracy in element identification.
# - Default to "No action" in ambiguous cases unless clear evidence ties changes to the task.
#     """
#     return prompt

def generate_vlm_action_prompt1(image_url1, json_file_1, image_url2, json_file_2, task_description, thought):
    prompt = f"""
You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. Your goal is to compare the screenshots, analyze changes in annotated interactive elements using both visual information and provided JSON descriptions, and infer the most likely actions based on the given context. The output should follow the output format, which includes the reasoning process (thought), the action type, and a natural language description of the action.

Inputs:

Screenshot 1: {image_url1}
JSON File 1: {json_file_1}
Screenshot 2: {image_url2}
JSON File 2: {json_file_2}
Task Description: {task_description}
Thought: {thought}

Analysis Steps:
1. Compare Screenshots Visually: Examine the provided screenshots to identify visual changes, focusing on annotated interactive elements (e.g., buttons, text fields).
2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
3. Incorporate Context: Use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
4. Infer Action Sequence: Determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

Action Space:
- click(start_box='<|box_start|>(x1,y1)<|box_end|>')
- left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
- right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
- drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
- hotkey(key='')
- type(content='')  # If you want to submit your input, use "\\" at the end of `content`.
- scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
- wait()  # Sleep for 5s and take a screenshot to check for any changes.
- finished()
- call_user()  # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

Output Format:
Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or ["No action"]]
Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or ["No action"]]

Guidelines:
- Use clear and user-friendly natural language.
- If no relevant changes or task-related changes occur, output "No action".
- If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
- Prioritize primary task-relevant actions when multiple changes are present.
- Rely explicitly on provided JSON data for accuracy in element identification.
- Default to "No action" in ambiguous cases unless clear evidence ties changes to the task.
    """
    return prompt

# def generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought):
#     prompt = f"""
# You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. You have been provided two screenshots separately (encoded in base64 format) along with their respective JSON descriptions.

# Your goal is to analyze the changes in annotated interactive elements using the provided JSON descriptions, visual differences observed from the provided screenshots, and infer the most likely actions based on the given context. Your output should follow the specified format, including the reasoning (thought), the identified actions, and natural language descriptions of each action.

# Inputs:

# Screenshot 1: [Provided separately as base64 encoded image]
# JSON File 1: {json_file_1}
# Screenshot 2: [Provided separately as base64 encoded image]
# JSON File 2: {json_file_2}
# Task Description: {task_description}
# Thought: {thought}

# Analysis Steps:
# 1. Visually Compare Screenshots: Identify visual changes by examining the separately provided screenshots, focusing on annotated interactive elements (e.g., buttons, text fields).
# 2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
# 3. Incorporate Context: Use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
# 4. Infer Action Sequence: Determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

# Action Space:
# click(start_box='<|box_start|>(x1,y1)<|box_end|>')
# left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
# right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
# drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
# hotkey(key='')
# type(content='') #If you want to submit your input, use \"\
# \" at the end of `content`.
# scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


# Output Format:
# Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
# Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or ["No action"]]
# Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or ["No action"]]

# Output Example:
# (Please follow this exact format for your output)

# {{
#     "Thought": "Upon comparing the two screenshots and their respective JSON files, I observed that the 'Username' field changed from empty to containing text. Additionally, the 'Login' button was clicked. The task description mentions logging in, and this matches the user typing a username and clicking the login button.",
#     "Actions": [
#         "type(content='john_doe')",
#         "click(start_box='(250,400)')"
#     ],
#     "Action NLP Descriptions": [
#         "Typed 'john_doe' into the 'Username' field",
#         "Clicked the 'Login' button at coordinates (250, 400)"
#     ]
# }}

# Guidelines:
# - Use clear and user-friendly natural language.
# - If no relevant changes or task-related changes occur, output "No action".
# - If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
# - Prioritize primary task-relevant actions when multiple changes are present.
# - Rely explicitly on provided JSON data for accuracy in element identification.
# - Default to "No action" in ambiguous cases unless clear evidence ties changes to the task.
#     """
#     return prompt

# def generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought):
#     prompt = f"""
# You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. You have been provided two screenshots separately (encoded in base64 format) along with their respective JSON descriptions.

# Your goal is to analyze the changes in annotated interactive elements using the provided JSON descriptions, visual differences observed from the provided screenshots, and infer the most likely actions based on the given context. Your output should follow the specified format, including the reasoning (thought), the identified actions, natural language descriptions of each action, and a strategic explanation of why these actions were taken.

# Inputs:

# Screenshot 1: [Provided separately as base64 encoded image]
# JSON File 1: {json_file_1}
# Screenshot 2: [Provided separately as base64 encoded image]
# JSON File 2: {json_file_2}
# Task Description: {task_description}
# Thought: {thought}

# Analysis Steps:
# 1. Visually Compare Screenshots: Identify visual changes by examining the separately provided screenshots, focusing on annotated interactive elements (e.g., buttons, text fields).
# 2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
# 3. Incorporate Context: Use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
# 4. Infer Action Sequence: Determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

# Action Space:
# click(start_box='<|box_start|>(x1,y1)<|box_end|>')
# left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
# right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
# drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
# hotkey(key='')
# type(content='') #If you want to submit your input, use \"\
# \" at the end of `content`.
# scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


# Output Format:
# Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
# Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or ["No action"]]
# Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or ["No action"]]
# Thought and Action NLP Descriptions: [Strategic explanation from the executor's perspective, explaining why these actions were taken based on the task, current observations, and chosen actions, e.g., "Since the task is to log in, and I observe a login form with empty username field on the screen, I should first type my username into the field and then click the login button to proceed."]

# Output Example:
# (Please follow this exact format for your output)

# {{
#     "Thought": "Upon comparing the two screenshots and their respective JSON files, I observed that the 'Username' field changed from empty to containing text. Additionally, the 'Login' button was clicked. The task description mentions logging in, and this matches the user typing a username and clicking the login button.",
#     "Actions": [
#         "type(content='john_doe')",
#         "click(start_box='(250,400)')"
#     ],
#     "Action NLP Descriptions": [
#         "Typed 'john_doe' into the 'Username' field",
#         "Clicked the 'Login' button at coordinates (250, 400)"
#     ],
#     "Thought and Action NLP Descriptions": "Since the task is to log into the system, and I can observe a login form with empty username field on the current screen, I need to first enter my credentials. I typed 'john_doe' in the username field, and then clicked on the 'Login' button at coordinates (250, 400) to submit my credentials and proceed with the authentication process."
# }}

# Guidelines:
# - Use clear and user-friendly natural language.
# - If no relevant changes or task-related changes occur, output "No action".
# - If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
# - Prioritize primary task-relevant actions when multiple changes are present.
# - Rely explicitly on provided JSON data for accuracy in element identification.
# - Default to "No action" in ambiguous cases unless clear evidence ties changes to the task.
# - For the "Thought and Action NLP Descriptions" field, put yourself in the position of the executor and explain your strategy by considering: what the task requires, what you need to accomplish in the current step, what you observe on the screen, and why you chose the specific actions.
#     """
#     return prompt

# def generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought):
#     prompt = f"""
# You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. You have been provided two screenshots separately (encoded in base64 format) along with their respective JSON descriptions.

# Your goal is to analyze the changes in annotated interactive elements using the provided JSON descriptions, visual differences observed from the provided screenshots, and infer the most likely actions based on the given context. You will first determine if meaningful changes have occurred, and if so, provide your output following the specified format, including the reasoning (thought), whether the frame is meaningful, the identified actions, natural language descriptions of each action, and a strategic explanation of why these actions were taken.

# Inputs:

# Screenshot 1: [Provided separately as base64 encoded image]
# JSON File 1: {json_file_1}
# Screenshot 2: [Provided separately as base64 encoded image]
# JSON File 2: {json_file_2}
# Task Description: {task_description}
# Thought: {thought}

# Analysis Steps:
# 1. Visually Compare Screenshots: Identify visual changes by examining the separately provided screenshots, focusing on annotated interactive elements (e.g., buttons, text fields).
# 2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
# 3. Determine Meaningfulness: Assess whether the detected changes are meaningful and relevant to the task at hand. If no meaningful changes are detected, mark the frame as not meaningful.
# 4. Incorporate Context: If the frame is meaningful, use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
# 5. Infer Action Sequence: For meaningful frames, determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

# Action Space:
# click(start_box='<|box_start|>(x1,y1)<|box_end|>')
# left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
# right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
# drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
# hotkey(key='')
# type(content='') #If you want to submit your input, use \"\
# \" at the end of `content`.
# scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


# Output Format:
# Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
# Meaningful: [Boolean value (true/false) indicating whether this frame contains meaningful actions or changes. If false, all subsequent outputs should be empty]
# Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or empty if not meaningful]
# Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or empty if not meaningful]
# Thought and Action NLP Descriptions: [Strategic explanation from the executor's perspective, explaining why these actions were taken based on the task, current observations, and chosen actions, e.g., "Since the task is to log in, and I observe a login form with empty username field on the screen, I should first type my username into the field and then click the login button to proceed.", or empty if not meaningful]

# Output Example:
# (Please follow this exact format for your output)

# {{
#     "Thought": "Upon comparing the two screenshots and their respective JSON files, I observed that the 'Username' field changed from empty to containing text. Additionally, the 'Login' button was clicked. The task description mentions logging in, and this matches the user typing a username and clicking the login button.",
#     "Meaningful": true,
#     "Actions": [
#         "type(content='john_doe')",
#         "click(start_box='(250,400)')"
#     ],
#     "Action NLP Descriptions": [
#         "Typed 'john_doe' into the 'Username' field",
#         "Clicked the 'Login' button at coordinates (250, 400)"
#     ],
#     "Thought and Action NLP Descriptions": "Since the task is to log into the system, and I can observe a login form with empty username field on the current screen, I need to first enter my credentials. I typed 'john_doe' in the username field, and then clicked on the 'Login' button at coordinates (250, 400) to submit my credentials and proceed with the authentication process."
# }}

# Example with non-meaningful frame:

# {{
#     "Thought": "After comparing the two screenshots and their JSON files, I didn't observe any significant changes related to the task. The screen elements remain in the same state and no user interaction appears to have occurred. This could be a loading state or a period where no action was taken.",
#     "Meaningful": false,
#     "Actions": [],
#     "Action NLP Descriptions": [],
#     "Thought and Action NLP Descriptions": ""
# }}

# Guidelines:
# - Use clear and user-friendly natural language.
# - If no relevant changes or task-related changes occur, set "Meaningful" to false and leave subsequent fields empty.
# - If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
# - Prioritize primary task-relevant actions when multiple changes are present.
# - Rely explicitly on provided JSON data for accuracy in element identification.
# - Set "Meaningful" to false in ambiguous cases unless clear evidence ties changes to the task.
# - For the "Thought and Action NLP Descriptions" field, put yourself in the position of the executor and explain your strategy by considering: what the task requires, what you need to accomplish in the current step, what you observe on the screen, and why you chose the specific actions.
# - The "Meaningful" field standardizes the judgment of whether the current frame is irrelevant or if no action has occurred. If "Meaningful" is false, all subsequent output fields should be empty arrays or empty strings.
#     """
#     return prompt

# def generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought):
#     prompt = f"""
# You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. You have been provided two screenshots separately (encoded in base64 format) along with their respective JSON descriptions.

# Your goal is to analyze the changes in annotated interactive elements using the provided JSON descriptions, visual differences observed from the provided screenshots, and infer the most likely actions based on the given context. You will first determine if meaningful changes have occurred, and if so, provide your output following the specified format, including the reasoning (thought), whether the frame is meaningful, the identified actions, natural language descriptions of each action, and a strategic explanation of why these actions were taken.

# Inputs:

# Screenshot 1: [Provided separately as base64 encoded image]
# JSON File 1: {json_file_1}
# Screenshot 2: [Provided separately as base64 encoded image]
# JSON File 2: {json_file_2}
# Task Description: {task_description}
# Thought: {thought}

# Analysis Steps:
# 1. Visually Compare Screenshots: Identify visual changes by examining the separately provided screenshots, focusing on annotated interactive elements (e.g., buttons, text fields).
# 2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
# 3. Determine Meaningfulness: Assess whether the detected changes are meaningful and relevant to the task at hand. If no meaningful changes are detected, mark the frame as not meaningful.
# 4. Incorporate Context: If the frame is meaningful, use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
# 5. Infer Action Sequence: For meaningful frames, determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

# Action Space:
# click(start_box='<|box_start|>(x1,y1)<|box_end|>')
# left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
# right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
# drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
# hotkey(key='')
# type(content='') #If you want to submit your input, use \"\
# \" at the end of `content`.
# scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


# Output Format:
# Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
# Meaningful: [Boolean value (true/false) indicating whether this frame contains meaningful actions or changes. If false, all subsequent outputs should be empty]
# Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or empty if not meaningful]
# Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or empty if not meaningful]
# Thought and Action NLP Descriptions: [CRUCIAL: This must be written ENTIRELY from the executor's first-person perspective, as if you are the person performing the task with NO knowledge of the annotation process. Do NOT mention coordinates, JSON files, screenshots, or any analysis - only what you can see on screen and what you're doing. Example: "I need to log in to continue, so I'm entering my username and clicking the login button to proceed." or empty if not meaningful]

# Output Example:
# (Please follow this exact format for your output)

# {{
#     "Thought": "Upon comparing the two screenshots and their respective JSON files, I observed that the 'Username' field changed from empty to containing text. Additionally, the 'Login' button was clicked. The task description mentions logging in, and this matches the user typing a username and clicking the login button.",
#     "Meaningful": true,
#     "Actions": [
#         "type(content='john_doe')",
#         "click(start_box='(250,400)')"
#     ],
#     "Action NLP Descriptions": [
#         "Typed 'john_doe' into the 'Username' field",
#         "Clicked the 'Login' button at coordinates (250, 400)"
#     ],
#     "Thought and Action NLP Descriptions": "I need to log into the system to complete my task. I can see a login form with username and password fields on the screen. For this step, I need to enter my credentials to gain access. I'll type 'john_doe' in the empty username field first, and then click on the login button to submit my information and continue with the authentication process."
# }}

# Example with non-meaningful frame:

# {{
#     "Thought": "After comparing the two screenshots and their JSON files, I didn't observe any significant changes related to the task. The screen elements remain in the same state and no user interaction appears to have occurred. This could be a loading state or a period where no action was taken.",
#     "Meaningful": false,
#     "Actions": [],
#     "Action NLP Descriptions": [],
#     "Thought and Action NLP Descriptions": ""
# }}

# Guidelines:
# - Use clear and user-friendly natural language.
# - If no relevant changes or task-related changes occur, set "Meaningful" to false and leave subsequent fields empty.
# - If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
# - Prioritize primary task-relevant actions when multiple changes are present.
# - Rely explicitly on provided JSON data for accuracy in element identification.
# - Set "Meaningful" to false in ambiguous cases unless clear evidence ties changes to the task.
# - For the "Thought and Action NLP Descriptions" field, you MUST write completely from the first-person perspective of someone performing the task, following a structured thinking pattern:
#   * CRITICAL: This field should NEVER contain any references to screenshots, JSON files, annotations, coordinates, or analysis.
#   * Structure the first-person narrative in a clear step-by-step thought process that includes:
#     1. Task goal: "I need to complete [overall task]..."
#     2. Current observation: "I can see [what's visible on screen]..."
#     3. Sub-task inference: "For this step, I need to [specific sub-task]..."
#     4. Action reasoning: "I'll [specific action with exact wording matching the Actions field]..."
#   * IMPORTANT: The actions described must match EXACTLY with the standardized actions listed in the "Actions" field. For example, if the Actions field has "type(content='john_doe')", the description should say "I'll type 'john_doe'" not "I'll enter my username"
#   * The person executing the task has no knowledge of the annotation process, screenshots, or JSON data - they only see what's on the screen.
#   * This should read like a natural first-person thought process of someone working through the task one step at a time.
# - The "Meaningful" field standardizes the judgment of whether the current frame is irrelevant or if no action has occurred. If "Meaningful" is false, all subsequent output fields should be empty arrays or empty strings.
#     """
#     return prompt

# def generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought):
#     prompt = f"""
# You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. You have been provided two screenshots separately (encoded in base64 format) along with their respective JSON descriptions.

# Your goal is to analyze the changes in annotated interactive elements using the provided JSON descriptions, visual differences observed from the provided screenshots, and infer the most likely actions based on the given context. You will first determine if meaningful changes have occurred, and if so, provide your output following the specified format, including the reasoning (thought), whether the frame is meaningful, the identified actions, natural language descriptions of each action, and a strategic explanation of why these actions were taken.

# Inputs:

# Screenshot 1: [Provided separately as base64 encoded image]
# JSON File 1: {json_file_1}
# Screenshot 2: [Provided separately as base64 encoded image]
# JSON File 2: {json_file_2}
# Task Description: {task_description}
# Thought: {thought}

# Analysis Steps:
# 1. Visually Compare Screenshots: Identify visual changes by examining the separately provided screenshots, focusing on annotated interactive elements (e.g., buttons, text fields).
# 2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
# 3. Determine Meaningfulness: Assess whether the detected changes are meaningful and relevant to the task at hand. If no meaningful changes are detected, mark the frame as not meaningful.
# 4. Incorporate Context: If the frame is meaningful, use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
# 5. Infer Action Sequence: For meaningful frames, determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

# Action Space:
# click(start_box='<|box_start|>(x1,y1)<|box_end|>') # x1,y1 are normalized coordinates between 0-1
# left_double(start_box='<|box_start|>(x1,y1)<|box_end|>') # x1,y1 are normalized coordinates between 0-1
# right_single(start_box='<|box_start|>(x1,y1)<|box_end|>') # x1,y1 are normalized coordinates between 0-1
# drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>') # x1,y1,x3,y3 are normalized coordinates between 0-1
# hotkey(key='')
# type(content='') #If you want to submit your input, use \"\
# \" at the end of `content`.
# scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left') # x1,y1 are normalized coordinates between 0-1
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


# Output Format:
# Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
# Meaningful: [Boolean value (true/false) indicating whether this frame contains meaningful actions or changes. If false, all subsequent outputs should be empty]
# Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or empty if not meaningful]
# Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or empty if not meaningful]
# Thought and Action NLP Descriptions: [CRUCIAL: This must be written ENTIRELY from the executor's first-person perspective, as if you are the person performing the task with NO knowledge of the annotation process. Do NOT mention coordinates, JSON files, screenshots, or any analysis - only what you can see on screen and what you're doing. Example: "I need to log in to continue, so I'm entering my username and clicking the login button to proceed." or empty if not meaningful]

# Output Example:
# (Please follow this exact format for your output)

# {{
#     "Thought": "Upon comparing the two screenshots and their respective JSON files, I observed that the 'Username' field changed from empty to containing text. Additionally, the 'Login' button was clicked. The task description mentions logging in, and this matches the user typing a username and clicking the login button.",
#     "Meaningful": true,
#     "Actions": [
#         "type(content='john_doe')",
#         "click(start_box='(0.5,0.8)')"
#     ],
#     "Action NLP Descriptions": [
#         "Typed 'john_doe' into the 'Username' field",
#         "Clicked the 'Login' button at coordinates (0.5, 0.8)"
#     ],
#     "Thought and Action NLP Descriptions": "I need to log into the system to complete my task. I can see a login form with username and password fields on the screen. For this step, I need to enter my credentials to gain access. I'll type 'john_doe' in the empty 'username' field first, and then click on the login button (0.5,0.8) to submit my information and continue with the authentication process."
# }}

# Example with non-meaningful frame:

# {{
#     "Thought": "After comparing the two screenshots and their JSON files, I didn't observe any significant changes related to the task. The screen elements remain in the same state and no user interaction appears to have occurred. This could be a loading state or a period where no action was taken.",
#     "Meaningful": false,
#     "Actions": [],
#     "Action NLP Descriptions": [],
#     "Thought and Action NLP Descriptions": ""
# }}

# Guidelines:
# - Use clear and user-friendly natural language.
# - If no relevant changes or task-related changes occur, set "Meaningful" to false and leave subsequent fields empty.
# - If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
# - Prioritize primary task-relevant actions when multiple changes are present.
# - Rely explicitly on provided JSON data for accuracy in element identification.
# - Set "Meaningful" to false in ambiguous cases unless clear evidence ties changes to the task.
# - For the "Thought and Action NLP Descriptions" field, you MUST write completely from the first-person perspective of someone performing the task, following a structured thinking pattern:
#   * CRITICAL: This field should NEVER contain any references to screenshots, JSON files, annotations, coordinates, or analysis.
#   * Structure the first-person narrative in a clear step-by-step thought process that includes:
#     1. Task goal: "I need to complete [overall task]..."
#     2. Current observation: "I can see [what's visible on screen]..."
#     3. Sub-task inference: "For this step, I need to [specific sub-task]..."
#     4. Action reasoning: "I'll [specific action with exact wording matching the Actions field]..."
#   * IMPORTANT: The actions described must match EXACTLY with the standardized actions listed in the "Actions" field. For example, if the Actions field has "type(content='john_doe')", the description should say "I'll type 'john_doe'" not "I'll enter my username"
#   * The person executing the task has no knowledge of the annotation process, screenshots, or JSON data - they only see what's on the screen.
#   * This should read like a natural first-person thought process of someone working through the task one step at a time.
# - The "Meaningful" field standardizes the judgment of whether the current frame is irrelevant or if no action has occurred. If "Meaningful" is false, all subsequent output fields should be empty arrays or empty strings.
#     """
#     return prompt

def generate_vlm_action_prompt(json_file_1, json_file_2, task_description, thought):
    prompt = f"""
You are a Visual Language Model (VLM) tasked with analyzing two consecutive screenshots from a task execution process to determine the actions that occurred between them. You have been provided two screenshots separately (encoded in base64 format) along with their respective JSON descriptions.

Your goal is to analyze the changes in annotated interactive elements using the provided JSON descriptions, visual differences observed from the provided screenshots, and infer the most likely actions based on the given context. You will first determine if meaningful changes have occurred, and if so, provide your output following the specified format, including the reasoning (thought), whether the frame is meaningful, the identified actions, natural language descriptions of each action, and a strategic explanation of why these actions were taken.

Inputs:

Screenshot 1: [Provided separately as base64 encoded image]
JSON File 1: {json_file_1}
Screenshot 2: [Provided separately as base64 encoded image]
JSON File 2: {json_file_2}
Task Description: {task_description}
Thought: {thought}

Analysis Steps:
1. Visually Compare Screenshots: Identify visual changes by examining the separately provided screenshots, focusing on annotated interactive elements (e.g., buttons, text fields).
2. Use JSON Files: Compare the element descriptions between JSON File 1 and JSON File 2 to detect changes in states (e.g., text fields changing content, buttons disappearing).
3. Determine Meaningfulness: Assess whether the detected changes are meaningful and relevant to the task at hand. If no meaningful changes are detected, mark the frame as not meaningful.
4. Incorporate Context: If the frame is meaningful, use the Task Description and Thought inputs to interpret the visual and JSON changes. Consider that multiple actions might occur between key frames.
5. Infer Action Sequence: For meaningful frames, determine the most likely sequence of actions based on the detected visual and JSON changes. If multiple actions are detected, provide the sequence in the output.

Action Space:
click(start_box='<|box_start|>(x1,y1)<|box_end|>') # x1,y1 are normalized coordinates between 0-1
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>') # x1,y1 are normalized coordinates between 0-1
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>') # x1,y1 are normalized coordinates between 0-1
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>') # x1,y1,x3,y3 are normalized coordinates between 0-1
hotkey(key='')
type(content='') #If you want to submit your input, use "\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left') # x1,y1 are normalized coordinates between 0-1
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


Output Format:
Thought: [Your reasoning process, explaining how you analyzed the screenshots, JSON files, task description, and thought to reach your conclusion]
Meaningful: [Boolean value (true/false) indicating whether this frame contains meaningful actions or changes. If false, all subsequent outputs should be empty]
Actions: [List of action types, e.g., ["click(start_box='<|box_start|>(x1,y1)<|box_end|>')", "type(content='username')"], or empty if not meaningful]
Action NLP Descriptions: [List of natural language descriptions of the actions, e.g., ["Clicked the 'Submit' button at coordinates (x1, y1)", "Typed 'john_doe' into the 'Username' field"], or empty if not meaningful]
Thought and Action NLP Descriptions: [CRUCIAL: This must be written ENTIRELY from the executor's first-person perspective, as if you are the person performing the task with NO knowledge of the annotation process. Do NOT mention coordinates, JSON files, screenshots, or any analysis - only what you can see on screen and what you're doing. MUST include visual descriptions of interactive elements, their approximate screen locations, and inferences about their functions. Example: "I need to log in to continue. I can see a white text input field labeled 'Username' in the center of the screen and a blue 'Login' button below it on the right side. The username field appears to be where I should enter my account name, while the blue button seems to be for submitting the login form. I'm entering my username in the text field and then clicking the blue login button to proceed." or empty if not meaningful]

Output Example:
(Please follow this exact format for your output)

{{
    "Thought": "Upon comparing the two screenshots and their respective JSON files, I observed that the 'Username' field changed from empty to containing text. Additionally, the 'Login' button was clicked. The task description mentions logging in, and this matches the user typing a username and clicking the login button.",
    "Meaningful": true,
    "Actions": [
        "type(content='john_doe')",
        "click(start_box='(0.5,0.8)')"
    ],
    "Action NLP Descriptions": [
        "Typed 'john_doe' into the 'Username' field",
        "Clicked the 'Login' button at coordinates (0.5, 0.8)"
    ],
    "Thought and Action NLP Descriptions": "I need to log into the system to complete my task. I can see a login form in the center of the screen with a white rectangular text input field labeled 'Username' and another one below it for 'Password'. There's a blue rectangular 'Login' button positioned at the bottom right of the form. The username field appears to be where I should enter my account name, while the blue button seems to be the submission button that will process my login credentials. For this step, I need to enter my credentials to gain access. I'll type 'john_doe' in the empty username field that's located in the middle of the form, and then click on the blue login button at the bottom right to submit my information and continue with the authentication process."
}}

Example with non-meaningful frame:

{{
    "Thought": "After comparing the two screenshots and their JSON files, I didn't observe any significant changes related to the task. The screen elements remain in the same state and no user interaction appears to have occurred. This could be a loading state or a period where no action was taken.",
    "Meaningful": false,
    "Actions": [],
    "Action NLP Descriptions": [],
    "Thought and Action NLP Descriptions": ""
}}

Guidelines:
- Use clear and user-friendly natural language.
- If no relevant changes or task-related changes occur, set "Meaningful" to false and leave subsequent fields empty.
- If multiple actions are detected between the two key frames, provide each action and its natural language description in sequence.
- Prioritize primary task-relevant actions when multiple changes are present.
- Rely explicitly on provided JSON data for accuracy in element identification.
- Set "Meaningful" to false in ambiguous cases unless clear evidence ties changes to the task.
- For the "Thought and Action NLP Descriptions" field, you MUST write completely from the first-person perspective of someone performing the task, following a structured thinking pattern:
  * CRITICAL: This field should NEVER contain any references to screenshots, JSON files, annotations, coordinates, or analysis.
  * Structure the first-person narrative in a clear step-by-step thought process that includes:
    1. Task goal: "I need to complete [overall task]..."
    2. Current observation with visual details: "I can see [detailed description of visible elements, their appearance (color, shape, size), and approximate locations on screen]..."
    3. Element function inference: "This [element] appears to be for [inferred function/purpose]..."
    4. Sub-task inference: "For this step, I need to [specific sub-task]..."
    5. Action reasoning with element descriptions: "I'll [specific action with exact wording matching the Actions field] on the [visual description of element] located [approximate screen position]..."
  * IMPORTANT: When describing interactive elements, include:
    - Visual appearance (color, shape, size, style)
    - Type of element (button, text field, dropdown, link, etc.)
    - Approximate location on screen (top/bottom, left/right/center, upper/lower portion, etc.)
    - Any visible labels or text on the element
    - Inference about the element's function or purpose based on its appearance and context
  * IMPORTANT: The actions described must match EXACTLY with the standardized actions listed in the "Actions" field. For example, if the Actions field has "type(content='john_doe')", the description should say "I'll type 'john_doe'" not "I'll enter my username"
  * The person executing the task has no knowledge of the annotation process, screenshots, or JSON data - they only see what's on the screen.
  * This should read like a natural first-person thought process of someone working through the task one step at a time while naturally describing what they see and interact with.
- The "Meaningful" field standardizes the judgment of whether the current frame is irrelevant or if no action has occurred. If "Meaningful" is false, all subsequent output fields should be empty arrays or empty strings.
    """
    return prompt

def construct_prompt_desktop(task, thought):
    """构造用于 UI 交互的 prompt"""
    return f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\
\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use Chinese in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{thought}
"""
