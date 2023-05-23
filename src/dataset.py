from .imports import *
from .env import *
from .config import *

data_folder = Path("../data")
en_ru_data_path = data_folder / "en-ru.txt"
if not en_ru_data_path.exists():
    data_folder.mkdir(exist_ok=True)
    wget.download(
        "https://raw.githubusercontent.com/neychev/made_nlp_course/master/datasets/Machine_translation_EN_RU/data.txt",
        out=str(en_ru_data_path),
    )

def dataset_convert_csv():
    with open(en_ru_data_path, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        en, ru = line.split("\t")
        yield dict(translation=dict(en=en, ru=ru))

test = Dataset.from_generator(
    dataset_convert_csv, cache_dir=os.environ["HF_DATASETS_CACHE"]
).train_test_split(0.1, seed=0)
val = test["test"].train_test_split(0.99, seed=0)

def create_dataset(
    enc_tokenizer: AutoTokenizer, dec_tokenizer: AutoTokenizer | None = None
) -> Dataset:
    if dec_tokenizer is None:
        dec_tokenizer = enc_tokenizer

    def preprocess_function(batch):
        inputs = enc_tokenizer(
            [(prefix + ex[source_lang]).strip() for ex in batch["translation"]]
        )
        outputs = dec_tokenizer(
            [ex[target_lang].strip() for ex in batch["translation"]]
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        batch["input_ids_length"] = [len(row) for row in batch["input_ids"]]
        batch["labels_length"] = [len(row) for row in batch["labels"]]

        return batch

    dataset = DatasetDict(
        dict(train=test["train"], val=val["train"], test=val["test"])
    ).map(preprocess_function, batched=True, batch_size=batch_size)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return dataset
