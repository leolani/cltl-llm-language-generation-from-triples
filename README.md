# cltl-llm-language-generation-from-triples

Part of the [Leolani](https://github.com/leolani) social robot framework. This
component turns structured output from Leolani's episodic knowledge graph
(eKG) — statements, thoughts, questions/answers and knowledge gaps expressed
as subject-predicate-object triples — into natural language replies, using a
local LLM through [Ollama](https://ollama.com) and
[LangChain](https://www.langchain.com/).

Instead of hand-written phrase templates, an instruction prompt is built for
each triple/thought and sent to the LLM, which paraphrases it into a short,
natural reply in the target language.

## How it works

- **`src/cltl/reply_generation/llm_triple_replier.py`** — `LLMTripleReplier`,
  the entry point. It configures a `ChatOllama` model and a `PromptProcessor`,
  reads a JSON file of eKG responses (see `data/`), and for each response
  generates and prints the prompts and the model's replies.
- **`src/cltl/reply_generation/prompts/response_processor.py`** —
  `PromptProcessor` extracts the relevant text (statement, thought, novelty,
  conflict, subject/complement knowledge gaps, etc.) from an eKG response and
  pairs it with the matching instruction to build the LLM prompt.
- **`src/cltl/reply_generation/prompts/instruct.py`** — `Instruct` holds the
  system prompts (instructions) used to tell the LLM how to paraphrase each
  kind of input (statement, answer, no-answer, subject gap, object gap,
  novelty, conflict) in a given language.

## Data

`data/` contains example eKG responses used as input/test fixtures, and a
notebook that walks through the generation process:

- `thoughts-responses.json`, `basic-statements-responses.json`,
  `basic-questions-responses.json`, `basic-mentions-responses.json`,
  `basic-experiences-responses.json`, `carl-responses.json`,
  `question_response.json` — sample eKG responses (statements, thoughts,
  questions, mentions, experiences) as produced by the Leolani brain.
- `test_response.py` — a single hardcoded example response used for testing.
- `Llama3-ThoughtsLangChain.ipynb` — a notebook demonstrating reply
  generation with `ChatOllama` and Llama 3.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with the desired model pulled
  (e.g. `ollama pull llama3.2`)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the replier against a sample data file:

```bash
python src/cltl/reply_generation/llm_triple_replier.py
```

This loads `data/thoughts-responses.json`, generates the prompts for each
response and prints the LLM's paraphrased replies to the console.

To use `LLMTripleReplier` in your own code:

```python
from cltl.reply_generation.llm_triple_replier import LLMTripleReplier

replier = LLMTripleReplier(language="Dutch", model_name="llama3.2")
prompts = replier._processor.get_all_prompt_input_from_response(response)
for prompt in prompts:
    reply = replier._ollama_client.invoke(prompt)
    print(reply.content)
```

## License

MIT — see [LICENSE](LICENSE).
