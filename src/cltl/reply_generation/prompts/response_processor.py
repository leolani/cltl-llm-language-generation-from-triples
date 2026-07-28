from cltl.reply_generation.prompts.instruct import Instruct

# * _statement_novelty
# * _entity_novelty
# * _negation_conflicts
# * _complement_conflict
# * _subject_gaps
# * _complement_gaps
# * _overlaps
# * _trust


class PromptProcessor():

    def __init__(self, language="English"):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """


        self._instruct = Instruct(language)

    def get_no_answer_prompt(self, question):
        prompt = [self._instruct.get_instruct_for_no_answer(), {"role": "system", "content": "I have no answer for this question"+question["utterance"]}]
        return prompt

    def get_answer_prompt(self, question, answer):
        prompt = [self._instruct.get_instruct_for_answer(), {"role": "system", "content": answer}]
        return prompt

    def get_thought_prompt(self, statement, thought):
        prompt = [self._instruct.get_instruct_for_statement(), {"role": "system", "content": statement+thought}]
        return prompt

    def get_utterance_from_statement (self, statement):
        if "utterance" in statement:
            utterance = statement["utterance"]
        else:
            utterance = ""
        return utterance

    def get_triple_text_from_statement (self, statement):
        triple = statement["triple"]
        triple_text = triple["_subject"]["_label"]+", "+triple["_predicate"]["_label"]+", "+triple["_complement"]["_label"]
        return triple_text

    def get_perspective_from_statement  (self, statement):
        perspective = statement["perspective"]
        perspective_text = ""
        if not perspective['_certainty']=='UNDERSPECIFIED':
            perspective_text += perspective['_certainty'].tolower()+ ", "
        if perspective['_polarity']=='POSITIVE':
            perspective_text += 'believes'+ ", "
        elif perspective['_polarity']=='NEGATIVE':
            perspective_text += 'denies'+ ", "
        if not perspective['_sentiment']=='NEUTRAL':
            perspective_text += perspective['_sentiment']+ ", "
        if not perspective['_emotion']=='UNDERSPECIFIED':
            perspective_text += perspective['_emotion'].tolower()+ ", "
        return perspective_text

    def get_author_from_statement (self, statement):
        author = statement["author"]["label"]
        return author

    #### In the case of a GAP, there is a trigger triple in the eKG that contains a "known entity".
    #### If the known entity is the subject of the trigger triple it can either be the subject or the complement (object) in a triple with another predicate.
    #### If a subject_gap_subject the known entity is subject in both the trigger triple and the unknown triple.
    #### If a subject_gap_complement the known entity is subject in the trigger triple and the object in the unknown triple.
    #### If a complement_gap_complement  the known entity is the complement in both the trigger triple and the unknown triple.
    #### If a complement_gap_subject the known entity is object in the trigger triple and the subject in the unknown triple.

    def get_subject_gap_subject(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text = known_entity +", "+predicate+", "+ gap_type[0]
        return gap_text

    def get_subject_gap_complement(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text = gap_type[0]+", "+predicate+", "+known_entity
        return gap_text

    def get_complement_gap_subject(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text =gap_type[0]+", ", predicate+", "+known_entity
        return gap_text

    def get_complement_gap_complement(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text = known_entity +", " + predicate+ ", " + gap_type[0]
        return gap_text

    def get_negation_conflict(self, conflict):
        provenance = conflict["_provenance"]
        author = provenance["_author"]["_label"]
        date = provenance["_date"]
        value = conflict["_polarity_value"]
        if value == "NEGATIVE":
            value = "denies"
        elif value == "POSITIVE":
            value = "claims"
        novelty_text = author + " "+ value+ " this on " + date
        return novelty_text


    # def get_cardinality_conflict(self, thought):
    #     novelty_text = author + " "+ value+ " me on " + date
    #     return novelty_text

    def get_provenance_from_statement_novelty(self, provenance):
        # {'_provenance': {
        #     '_author': {'_id': 'http://cltl.nl/leolani/friends/carl', '_label': 'carl', '_offset': None, '_confidence': 0.0,
        #                 # '_types': ['Source', 'Actor']}, '_date': '2017-10-24'}}
        author = provenance["_author"]["_label"]
        date = provenance["_date"]
        novelty_text = author + " told me this on "+ date
        return novelty_text

    def get_all_prompt_input_from_response(self, response):
        statement = response["statement"]
        statement_text = self.get_triple_text_from_statement(statement)
        statement_author = self.get_author_from_statement(statement)
        prompts = []
        thought = response["thoughts"]
        #print("THOUGHT", thought)
        novelties = thought["_statement_novelty"]
      #  print("novelties", novelties)
        for novelty in novelties:
            input = statement_author + " claims " + statement_text + ". Also " + self.get_provenance_from_statement_novelty(novelty["_provenance"])
            prompt = [self._instruct.get_instruct_for_novelty(), {"role": "user", "content": input}]
            prompts.append(prompt)

        # @TODO
        # novelties = thought["_entity_novelty"]
        # for novelty in novelties:
        #     input = statement_author+" claims "+ statement_text+". Also "+get_XXX_from_entity_novelty(novelty)
        #     prompt = [instruct.instruct_for_novelty, {"role": "user", "content": input}]
        #     prompts.append(prompt)

        conflicts = thought["_negation_conflicts"]
      #  print("conflicts", conflicts)
        for conflict in conflicts:
            input = statement_author + " claims " + statement_text + ". But " + self.get_negation_conflict(conflict)
            prompt = [self._instruct.get_instruct_for_novelty(), {"role": "user", "content": input}]
            prompts.append(prompt)

        # @TODO
        # conflicts = thought["_complement_conflict"]
        # for conflict in conflicts:
        #     input = statement_author+" claims "+ statement_text+". Also "+get_complement_conflict(conflict)
        #     inputs.append(input)

        gaps = thought["_subject_gaps"]
       # print("_subject_gaps", gaps)
        if gaps["_subject"]:
            for gap in gaps["_subject"]:
                input = self.get_subject_gap_subject(gap)
                prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
             #   print(prompt)
                prompts.append(prompt)

            if gaps["_complement"]:
                for gap in gaps["_complement"]:
                    input = self.get_subject_gap_complement(gap)
                    prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
              #      print(prompt)

                    prompts.append(prompt)

        gaps = thought["_complement_gaps"]
       # print("_complement_gaps", gaps)

        if gaps["_subject"]:
            for gap in gaps["_subject"]:
                input = self.get_complement_gap_subject(gap)
                prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
               # print(prompt)

                prompts.append(prompt)

            if gaps["_complement"]:
                for gap in gaps["_complement"]:
                    input = self.get_complement_gap_complement(gap)
                    prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
                #    print(prompt)

                    prompts.append(prompt)

        # @TODO
        # overlaps = thought["_overlaps"]
        # for overlap in overlaps:
        #     input = statement_author+" claims "+ statement_text+". Also "+get_overlap_statement(overlap)
        #     inputs.append(input)

        # @TODO
        # trust = thought["_trust"]
        # input = statement_author+" claims "+ statement_text+". Also "+get_trust_statement(trust)
        # inputs.append(input)
        return prompts
