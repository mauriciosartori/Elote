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
    number_of_sentence_to_evaluate = 3

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

    def translate_sentences_es_to_en(self):
        print("Translating sentences from Spanish to English...")
        list_src_es_sentences = self.get_list_sentences('resources/base_tedx_spanish_corpora.txt')

        model_name = 'Helsinki-NLP/opus-mt-es-en'
        list_trans_sentences_es_en = translate_from_MarianModel(model_name, list_src_es_sentences)
        write_sentences_in_file(list_trans_sentences_es_en, 'resources/generated_tedx_english_corpora.txt')
        print("Translation done")

    def translate_sentences_en_to_es(self):
        print("Translating sentences from English to Spanish...")
        list_src_en_sentences = self.get_list_sentences('resources/generated_tedx_english_corpora.txt')

        model_name = 'Helsinki-NLP/opus-mt-en-es'
        list_trans_sentences_en_es = translate_from_MarianModel(model_name, list_src_en_sentences)
        write_sentences_in_file(list_trans_sentences_en_es, 'resources/results_tedx_spanish_corpora.txt')
        print("Translation done")

    def evaluate_results(self):
        # In order to do this evaluation, we have to compare the sentences
        # that are the result of the translation EN-ES with the original es corpus

        # List of sentences from base ES corpus
        list_base_es_sentences = self.get_list_sentences('resources/base_tedx_spanish_corpora.txt')

        # List of sentences from the ES result set
        list_results_es_sentences = self.get_list_sentences('resources/results_tedx_spanish_corpora.txt')

        # We compare each sentence and only perform analysis if it is different
        for index in range(0, len(list_base_es_sentences), 1):
            if list_base_es_sentences[index] == list_results_es_sentences[index]:
                print("This string ins the same")
            else:
                print(list_base_es_sentences[index])
                print(list_results_es_sentences[index])
                print("-----------------------")
