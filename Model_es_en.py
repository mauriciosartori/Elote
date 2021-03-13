from transformers import MarianMTModel, MarianTokenizer


class Model_es_en:
    def print_hola(self):
        src_text = [
            '>>fra<< En donde estas',
            '>>por<< La doctora comio su lunch',
            '>>esp<< Hay amiga!'
        ]

        # model_name = 'Helsinki-NLP/opus-mt-en-es'
        model_name = 'Helsinki-NLP/opus-mt-es-en'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        print("tgt_text")
        print(tgt_text)

