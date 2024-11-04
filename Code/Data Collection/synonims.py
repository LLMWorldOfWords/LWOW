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
        profiles[f"{data['cue'].lower()} {data['context']}"] = None


with open("mistral_synonims.jsonl", "w") as f:
    for profile in tqdm.tqdm(profiles):
        

        prompt = f"""
                You are partecipating in a scientific study about the human language. 

                Task:
                 - You will be provided with an input word and a specific context (within parentheses): write the word synonims related to the specified context.

                Example: 
                Input: bank [finance]
                Output: treasury * storehouse * vault
                 
                 Constraints:
                 - Separe consecutive synonims with an asterisk (*). Other separators will be considered incorrect.
                 - Avoid enumerating the synonims.
                 - Do not explain the synonims nor the task.
                 - Each synonim should be a single word.
                 - Additional text in the response will be considered a severe error.
                 - Do not use new lines.
                 - Do not use the terms "Input" and "Output" in the response.
                 - In case of no synonims, leave the response empty.
                 
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

        #for word in profile:

        results = []

        u1.initiate_chat(
                u2,
                message=f"""{profile}""",
                silent=True,  # default is False
                max_round=0,  # default is 3
        )
        cleaned = True
        out = u1.chat_messages[u2][-1]["content"]

        
        out = out.replace("\n", "*").replace("~", "*").replace(";", "*").replace("Input:", "").replace("Output:", "").replace(".", "").lstrip().rstrip()
        pattern_order = r'[0-9]'
        out = re.sub(pattern_order, '', out)

        if "*" in out:
            out = [x.lstrip().rstrip() for x in out.split("*")]
            out = [re.sub(r'\W+', '', x) for x in out if 0 < len(x) <= 10]
        elif "," in out:
            out = [x.lstrip().rstrip() for x in out.split(",")]
            out = [re.sub(r'\W+', '', x) for x in out if 0 < len(x) <= 10]
        else:
            if len(out) <= 10:
                out = [re.sub(r'\W+', '', out)]
            else:
                out = []
        

        profile = profile.split("[")

        res = {
                    "cue": profile[0].rstrip(),
                    "context": f"[{profile[1]}",
                    "synonims": out
            }

        f.write(json.dumps(res) + "\n")
        f.flush()
