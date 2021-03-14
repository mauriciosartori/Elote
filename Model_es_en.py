from transformers import MarianMTModel, MarianTokenizer


class Model_es_en:

    def get_list_sentences_es(self, corpora_name):
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

    def tranlate_sentences_es_to_engl(self):
        list_src_es_sentences = self.get_list_sentences_es('resources/base_tedx_spanish_corpora.txt')
        print(list_src_es_sentences)

        # model_name = 'Helsinki-NLP/opus-mt-en-es'
        model_name = 'Helsinki-NLP/opus-mt-es-en'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(list_src_es_sentences, return_tensors="pt"))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        print("tgt_text")
        print(tgt_text)
        print("Translation done")
