import json

from langchain_ollama import ChatOllama
from cltl.reply_generation.prompts.response_processor import PromptProcessor

class LLMTripleReplier():
    def __init__(self, language="English", model_name = "llama3.2",):
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """
        self._language = language
        self._ollama_client = ChatOllama( model= model_name,  temperature=0.1, num_ctx=4096 )    # limits KV-cache size, avoids GPU OOM
        self._processor = PromptProcessor(language)


if __name__ == "__main__":
    model_name = "llama3.2"
    replier = LLMTripleReplier(language="Dutch", model_name=model_name)
    path = "../../../data/thoughts-responses.json"
    print(path)
    file = open(path)
    data = json.load(file)
    for response in data:
        prompts = replier._processor.get_all_prompt_input_from_response(response)
        new_message = ""
        for prompt in prompts:
            print("PROMPT", prompt)
            response = replier._ollama_client.invoke(prompt)
            print('RESPONSE:', response.content)



