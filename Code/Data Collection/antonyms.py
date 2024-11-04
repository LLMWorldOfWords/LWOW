import json
import tqdm
import re
import random
from autogen import AssistantAgent

config_list = [
    {
        "model": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "api_base": "",
        "api_type": "open_ai",
        "api_key": "NULL",
    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
    "request_timeout": 1200,
}

profiles = {}

with open("mistral_wordnet.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        profiles[f"{data['cue'].lower()}"] = None

with open("mistral_antonyms_new.jsonl", "w") as f:
    for profile in tqdm.tqdm(profiles):
        

        prompt = f"""
                You are participating in a scientific study about the human language. 

                Definition:
                - An antonym is a word that expresses a meaning opposed to the meaning of another word. Example: The word "happy" has the antonym "sad".

                Task:
                 - You will be provided with an input word: write up to 5 antonyms for the input word.
                 - In case the input word does not have antonyms, answer with *. 
                 
                Example: 
                Input: happy 
                Output: sad * unhappy

                Input: dog
                Output: *
                 
                 Constraints:
                 - Separate consecutive antonyms with an asterisk (*). Other separators will be considered incorrect.
                 - Do not explain the antonym or the task.
                 - Additional text in the response will be considered a severe error.
                 - Misleading or wrong responses will be considered a severe error. 
                """

        u2 = AssistantAgent(
            name=f"Agent",
            llm_config=llm_config,
            system_message=prompt,
            max_consecutive_auto_reply=1,
        )

        u1 = AssistantAgent(
            name=f"Agent 1",
            system_message="You are an agent that writes a single word at a time",
            llm_config=llm_config,
            max_consecutive_auto_reply=0,
        )

        results = []

        u1.initiate_chat(
                u2,
                message=f"""{profile}""",
                silent=True,  # default is False
                max_round=0,  # default is 3
        )
        cleaned = True
        out = u1.chat_messages[u2][-1]["content"]
        
        out = out.replace("\n", "*").replace("~", "*").replace("Input:", "").replace("Output:", "").lstrip().rstrip()

        if "*" in out:
            out = [x.lstrip().rstrip() for x in out.split("*")]
            out = [re.sub(r'\W+', '', x) for x in out if 0 < len(x) <= 10]
        elif "," in out:
            out = [re.sub(r'\W+', '', x).lstrip().rstrip() for x in out.split(",")]
            out = [re.sub(r'\W+', '', x) for x in out if 0 < len(x) <= 10]
        else:
            if 0 < len(out) <= 10:
                out = [re.sub(r'\W+', '', out)]
            else:
                out = []
        

        res = {
                    "cue": profile.rstrip(),
                    "antonyms": out
            }

        f.write(json.dumps(res) + "\n")
        f.flush()
