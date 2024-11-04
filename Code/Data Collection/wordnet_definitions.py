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
        "max_tokens": 1000000,
    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
    "request_timeout": 1200,
}

profiles = json.load(open("cues_profiles.json"))
cues = {}
for profile in profiles:
    for word in profile[1]:
       cues[word] = None

profiles = list(cues.keys())

random.shuffle(profiles)

with open("mistral_wordnet_2.jsonl", "w") as f:
    for profile in tqdm.tqdm(profiles):
        

        prompt = f"""
                Task:
                 - You will be provided with an input word: report all its definitions (separed by an asterisk) specifying their meanings.
                 - Always specify the meanings of the definition within parentheses.
                 
                 Constraints:
                 - Avoid enumerating the definitions.
                 - Separe context and definition with an at sign (@).
                 - Separe consecutive definitions with an asterisk (*).
                 - Definitions without a specified meaning will be considered incorrect.
                 - Definitions composed by less than 5 words will be considered incorrect.
                 - Definitions that do not end with a full stop will be considered incorrect.
                
                Example: 
                - Input: 'bank'
                - Output: [finance] @ A financial institution that accepts deposits, makes loans, and performs various other financial services for individuals and businesses. * [sport] @ In cricket, to hit the ball in a way that causes it to bounce off the ground before reaching the fielders.
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
                message=f"""'{profile}'""",
                silent=True,  # default is False
                max_round=0,  # default is 3
        )
        cleaned = True
        out = u1.chat_messages[u2][-1]["content"]
        
        out = out.split("*")

        for o in out:

            o = o.lstrip()

            if len(o) < 1 or o[0] != "[":
                continue

            o = o.split("@")
            if len(o) < 2:
                continue

            context = o[0].strip()
            definition = o[1].strip()

            definition = definition.split("\n")[0]

            if len(definition) < 0:
                continue

            res = {
                    "cue": profile.lower(),
                    "context": context,
                    "definition": definition
            }

            f.write(json.dumps(res) + "\n")
        f.flush()
