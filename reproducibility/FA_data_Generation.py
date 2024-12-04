import json
import tqdm
import re
from autogen import AssistantAgent
from FA_Functions import *

# update based on LLM
config_list = [
    {
        "model": "mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
        "api_base": "http://10.8.0.1:8081/v1",
        "api_type": "open_ai",
        "api_key": "NULL",
    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
    "request_timeout": 1200,
}

# Load SWOW data
simplifiedSWOW = loadSimplifiedSWOW()
profiles_cues = getProfilesCues(simplifiedSWOW)

words = [word for profiles_cues in profiles_cues for word in profiles_cues[1]]

# update based on LLM
with open('./data/original_datasets/mistral_free_associations.jsonl', "w") as f:
    for word in tqdm.tqdm(words):
        

        prompt = f""" 
                Task:
                 - You will be provided with an input word: write the first 3 words you associate to it separated by a comma.
                 - No additional output text is allowed. 
                
                Constraints:
                - no carriage return characters are allowed in the answers.
                - answers should be as short as possible.
                            
                Example: 
                Input: sea
                Output: water,beach,sun
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
            message=f"""{word}""",
            silent=True,  # default is False
            max_round=0,  # default is 3
        )
        cleaned = True
        out = u1.chat_messages[u2][-1]["content"]

        # output cleaning
        try:
            if len(out) > 0:
                out = [u[1:].strip() if len(u) > 1 and u[0] == ' ' else u.strip() for u in out.split(",")]
                r = []
                for a in out:
                    a = a.lower().split(" ")
                    if len(a) <= 2:
                        a = [re.sub(r'\W+', '', x) for x in a]
                        a = [x for x in a if len(x) > 1]
                        r.append(" ".join(a))
                out = r

            else:
                out = []
        except:
            cleaned = False

        res = {
                "cue": word.lower(),
                "response": out,
                'cleaned': cleaned,
                
            }

        f.write(json.dumps(res) + "\n")
        f.flush()
