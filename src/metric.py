from IPython.display import display
from .imports import *
from .config import *

metric = evaluate.load("bleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(tokenizer, metric, eval_preds):
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]}
    result['gen_len'] = np.mean([len(p) for p in decoded_preds])
    
    return result

def compute_metrics_test(model, collator, tokenizer, metric, batch, log=False):
    data = collator(dict(input_ids=batch["input_ids"]))

    preds = model.generate(
        data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
    ).cpu()
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    source = [s[source_lang] for s in batch["translation"]]
    decoded_labels = [s[target_lang] for s in batch["translation"]]
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric.add_batch(references=decoded_labels, predictions=decoded_preds)
    if log:
        display(
            [
                dict(input=i, target=l, preds=p)
                for p, l, i in zip(decoded_preds, decoded_labels, source)
            ]
        )
