import cltl.reply_generation.prompts.response_processor as processor

# _statement_novelty
# _entity_novelty
# _negation_conflicts
# _complement_conflict
# _subject_gaps
# _complement_gaps
# _overlaps
# _trust

class Instruct():

    def __init__(self, language="English"):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """
        self._language = language

    def get_instruct_for_statement (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a phrase, followed by a perspective, followed by \"that\" and a triple with a subject, predicate and object.\
         You need to paraphrase the input in plain " + self._language + ". \
         Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
        }
        return instruct

    def get_instruct_for_answer (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: the question, followed by the answer.\
         You need to paraphrase the input in plain " + self._language + ". \
         Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
        }
        return instruct

    def get_instruct_for_no_answer (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         You were given a question but you do not have an answer from your memory.\
         You are given the question as input and need to explain that you have no answer in plain " + self._language + ". \
         Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         The response should be just the paraphrased text and nothing else."
        }
        return instruct


    def get_instruct_for_subject_gap (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of subject.\
         You need to paraphrase the input in plain " + self._language + " as a question for the object. \
         Use who for the type person, where for the type location, when for the type time and what for everything else. \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct


    def get_instruct_for_object_gap (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of object.\
         You need to paraphrase the input in plain  " + self._language + "  as a question for the object. \
         Use who for the type person, where for the type location, when for the type time and what for everything else. \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct

    def get_instruct_for_novelty (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of object and a phrase.\
         You need to paraphrase the input in plain  " + self._language + ". \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct

    def get_instruct_for_conflict (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of object and a phrase that expresses a conflict.\
         You need to paraphrase the input in plain  " + self._language + ". \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
           }
        return instruct
