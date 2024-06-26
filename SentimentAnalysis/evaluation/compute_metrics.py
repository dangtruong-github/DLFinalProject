from datasets import load_metric

metrics_list = load_metric('glue', 'mrpc', trust_remote_code=True)


def compute_metrics(config, eval_pred):
    predictions, labels = eval_pred
    print(predictions)
    return metrics_list.compute(predictions=predictions, references=labels)
