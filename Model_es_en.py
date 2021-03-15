from transformers import MarianMTModel, MarianTokenizer


class Model_es_en:

    def get_list_sentences(self, corpora_name):
        with open(corpora_name) as file:
            full_base_data_set = file.readlines()

        # We can control the number of sentences to work with
        number_of_sentence_to_translate = 3
        number_sentences_trained = 0
        list_es_sentences = []
        for sentence in full_base_data_set:
            number_sentences_trained += 1
            if number_sentences_trained == number_of_sentence_to_translate:
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
        list_trans_sentences_es_en = self.tranlate_from_MarianModel(model_name)
        self.write_en_sentences(list_trans_sentences_es_en)
        print("Translation done")

    def write_en_sentences(self, list_es_sentences):
        with open('resources/generated_tedx_english_corpora.txt', 'w') as file:
            file.write("\n".join(list_es_sentences))

    def translate_sentences_en_to_es(self):
        print("Translating sentences from English to Spanish...")
        list_src_en_sentences = self.get_list_sentences('resources/generated_tedx_english_corpora.txt')

        model_name = 'Helsinki-NLP/opus-mt-en-es'
        list_trans_sentences_es_en = self.tranlate_from_MarianModel(model_name)
        self.write_en_sentences(list_trans_sentences_es_en)
        print("Translation done")

    def tranlate_from_MarianModel(self, model_name):
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        # print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(list_src_es_sentences, return_tensors="pt"))
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
