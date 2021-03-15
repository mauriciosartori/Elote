from Marian_evaluator import Marian_evaluator

if __name__ == '__main__':
    model_es_en = Marian_evaluator()
    model_es_en.translate_sentences_es_to_en()
    model_es_en.translate_sentences_en_to_es()
    model_es_en.evaluate_results()