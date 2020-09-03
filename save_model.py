import argparse
import os

from transformers import  BertModel, BertTokenizer


def download_model(model_card, save_path):
    #model_card = "bert-base-cased"  #### Please refer https://huggingface.co/models for different version of BERT models
    model = BertModel.from_pretrained(model_card)
    tokenizer = BertTokenizer.from_pretrained(model_card)

    file_path = os.path.join(save_path, model_card)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    model.save_pretrained(file_path)
    tokenizer.save_pretrained(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Fine models from https://huggingface.co/models, and save it to the local directory'
    )

    parser.add_argument('--model', help='the model card', required=True)
    parser.add_argument('--dir', help='the saving directory', required=True)

    args = parser.parse_args()
    model = args.model
    dir = args.dir

    download_model(model, dir)
