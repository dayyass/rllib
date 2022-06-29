import torch

if __name__ == "__main__":

    # Russian to English translation
    ru2en = torch.hub.load(
        "pytorch/fairseq",
        "transformer.wmt19.ru-en",
        checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
        tokenizer="moses",
        bpe="fastbpe",
    )
    ru2en.translate("Машинное обучение - это здорово!")  # 'Machine learning is great!'
