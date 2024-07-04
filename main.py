import warnings

import pandas as pd
import torch

from datasets import datasets
from experiment import run_exp, ExperimentDto
from models import models
from result import Result
from sparsing.sparsing_algorithms import sparsing_list
import time

from email_sender import send_email

start_time = time.time()

warnings.filterwarnings('ignore')
device = torch.device('cpu')
print(f'Using device: {device}')


def get_model(model_type, dataset):
    try:
        model = model_type(dataset)
    except TypeError:
        model = model_type(dataset.num_node_features, dataset.num_node_features, 3, dataset.num_classes)
    return model.to(device)


if __name__ == '__main__':
    run_num = 10
    results = []
    for dataset in datasets:
        dataset_results = []
        for model_data in models:
            for sparsing in sparsing_list:
                sparsing_name = sparsing.__name__ if sparsing is not None else 'NoSparsification'
                for percent2remove in range(1, 9) if sparsing is not None else [None]:
                    sparsing_alg = sparsing if percent2remove is None else sparsing(percent2remove)
                    model = get_model(model_data.model_type, dataset)
                    experiment_dto = ExperimentDto(dataset, model, sparsing_alg)
                    run_results = [run_exp(experiment_dto, i) for i in range(run_num)]
                    acc, removed_percentages = zip(*run_results)
                    removed_percentage = removed_percentages[0]

                    result = Result(model_data.model_name, dataset.name, sparsing_name, acc, percent2remove,
                                    removed_percentage)
                    print(result)
                    results.append(result.as_dict())
                    dataset_results.append(result.as_dict())
        df = pd.DataFrame(dataset_results)
        df.to_csv(f'additional_files/{dataset.name}_results.csv', index=False)

        recipients = ['piotr.bledowski.77@gmail.com']
        subject = f'{dataset.name} results'
        body = f'The {dataset.name} results at run_num=100 are here!'
        attachment = f'additional_files/{dataset.name}_results.csv'

        send_email(recipients, subject, body, attachment)

    results_df = pd.DataFrame(results)
    results_df.to_csv('additional_files/results.csv', index=False)

end_time = time.time()
total_time = end_time - start_time
print(f'Total time: {total_time} seconds ({total_time / 3600.0} hours)')
