import json

from transformers import MarianMTModel, MarianTokenizer


def write_sentences_in_file(list_es_sentences, file):
    with open(file, 'w') as file:
        file.write("\n".join(list_es_sentences))


def translate_from_MarianModel(model_name, list_src_es_sentences):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # print(tokenizer.supported_language_codes)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(list_src_es_sentences, return_tensors="pt"))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


class Marian_evaluator:
    number_of_sentence_to_evaluate = 5000

    def get_list_sentences(self, corpora_name):
        with open(corpora_name) as file:
            full_base_data_set = file.readlines()

        # We can control the number of sentences to work with
        number_sentences_trained = 0
        list_es_sentences = []
        for sentence in full_base_data_set:
            number_sentences_trained += 1
            if number_sentences_trained == self.number_of_sentence_to_evaluate:
                break
            # In order to keep only text related to the sentence's context
            # We have to remove the TEDx postfix on each sentence
            formatted_sentence = sentence.rsplit(' ', 1)[0]
            list_es_sentences.append(formatted_sentence)
        return list_es_sentences

    def get_list_candidate_items_from_train(self):
        with open('resources/twt.train.json') as file:
            full_base_data_set = file.readlines()

        # We can control the number of sentences to work with
        number_sentences_trained = 0
        list_candidate_tokens = []

        for sentence_split_str in full_base_data_set:
            number_sentences_trained += 1
            if number_sentences_trained == self.number_of_sentence_to_evaluate:
                break
            sentence_split_json = json.loads(sentence_split_str)
            sentence = ""
            save_sentence = False
            for index in range(3, len(sentence_split_json), +1):
                single_word_compose = sentence_split_json[index]
                single_word = single_word_compose[0]
                if single_word.lower() == "the":
                    save_sentence = True
                sentence += " " + single_word
            if save_sentence:
                list_candidate_tokens.append(sentence)

        return list_candidate_tokens

    def generate_sentences_to_analyze(self):
        print("Getting list of sentences from train corpora...")
        list_src_es_sentences = self.get_list_candidate_items_from_train()
        print("Number of sentence found:" + str(len(list_src_es_sentences)))
        write_sentences_in_file(list_src_es_sentences, 'resources/candidate_sentences.txt')
        print("Sentences generated and saved")

    def translate_sentences_en_to_es(self):
        print("Translating sentences from English to Spanish...")
        list_src_en_sentences = self.get_list_sentences('resources/candidate_sentences.txt')
        sentences_with_the = []

        for sentence in list_src_en_sentences:
            if "The" in sentence or "the" in sentence:
                sentences_with_the.append(sentence)

        current_index = 0
        for group_index in range(0, 10, +1):
            sub_list_sentences = []
            for index in range(current_index, current_index + 70, +1):
                sub_list_sentences.append(sentences_with_the[current_index])
                current_index += 1
            model_name = 'Helsinki-NLP/opus-mt-en-es'
            list_trans_sentences_en_es = translate_from_MarianModel(model_name, sub_list_sentences)
            file_name = 'resources/' + str(current_index) + 'candidate_sentences_translated.txt'
            write_sentences_in_file(list_trans_sentences_en_es, str(file_name))

        print("Translation done")

    def evaluate_results(self):

        # List of sentences from the ES result set
        list_results_es_sentences = self.get_list_sentences('resources/candidate_sentences_translated.txt')
        list_male_genre_sentences = []
        for sentence in list_results_es_sentences:
            if " el" in sentence \
                    or "El " in sentence \
                    or "La " in sentence \
                    or " la " in sentence \
                    or "Los " in sentence:
                start_index = sentence.find(" el ")
                if start_index == -1:
                    start_index = sentence.find("El ")
                if start_index == -1:
                    start_index = sentence.find("La ")
                if start_index == -1:
                    start_index = sentence.find(" la ")
                if start_index == -1:
                    start_index = sentence.find("Los ")
                if start_index != -1 and len(sentence) > start_index + 5:
                    word_context = sentence[start_index:]
                    sentence_array = word_context.split()
                    article_noun_sentence = sentence_array[0] + " " + sentence_array[1] + " - " + sentence
                    list_male_genre_sentences.append(article_noun_sentence)

        write_sentences_in_file(list_male_genre_sentences, 'resources/results_male_genre_sentences.txt')
        print("-----------------------")
