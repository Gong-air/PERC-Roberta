import os
import sys
sys.path.append(".")

import csv
import json
import argparse

from openprompt.data_utils import InputExample
from openprompt.trainer import ClassificationRunner, GenerationRunner
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from openprompt.protoverb_trainer import ProtoVerbClassificationRunner
from re import template
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.prompts import load_template, load_verbalizer, load_template_generator, load_verbalizer_generator
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.config import get_config, save_config_to_yaml
from openprompt.plms import load_plm_from_config
from openprompt.data_utils import load_dataset
from openprompt.utils.cuda import model_to_device
from openprompt.data_utils.data_processor import DataProcessor
from yacs.config import CfgNode



def build_dataloader(dataset, template, tokenizer,tokenizer_wrapper_class, config, split):
    dataloader = PromptDataLoader(
        dataset = dataset,
        template = template,
        tokenizer = tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        batch_size = config[split].batch_size,
        shuffle = config[split].shuffle_data,
        teacher_forcing = config[split].teacher_forcing if hasattr(config[split],'teacher_forcing') else None,
        predict_eos_token = True if config.task == "generation" else False,
        **config.dataloader
    )
    return dataloader

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x

class ERCProcessor(DataProcessor):
    """
    `AG News <https://arxiv.org/pdf/1509.01626.pdf>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/yumeng5/LOTClass>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "agnews"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ['neutral', 'surprise', 'fear', 'sadness','joy','disgust','anger']

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label))
                examples.append(example)
        return examples
def load_ERCdataset(config: CfgNode, return_class=True, test=False):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.

    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    dataset_config = config.dataset

    processor = ERCProcessor()

    train_dataset = None
    valid_dataset = None
    if not test:
        try:
            train_dataset = processor.get_train_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no training dataset in {dataset_config.path}.")
        try:
            valid_dataset = processor.get_dev_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no validation dataset in {dataset_config.path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(dataset_config.path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {dataset_config.path}.")
    # checking whether donwloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset





def main():
    config, args = get_config()
    # exit()
    # init logger, create log dir and set log level, etc.
    if args.resume and args.test:
        raise Exception("cannot use flag --resume and --test together")
    if args.resume or args.test:
        config.logging.path = EXP_PATH = args.resume or args.test
    else:
        EXP_PATH = config_experiment_dir(config)
        init_logger(os.path.join(EXP_PATH, "log.txt"), config.logging.file_level, config.logging.console_level)
        # save config to the logger directory
        save_config_to_yaml(config)


    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_ERCdataset(config, test = args.test is not None or config.learning_setting == 'zero_shot')

    # train_dataset_original = json.load(open("dataset_own/meld_train.json"))
    # train_dataset_label_original = json.load(open("dataset_label/meld_train.json"))
    # train_dataset_original = jsonKeys2int(train_dataset_original)
    # train_dataset_label_original = jsonKeys2int(train_dataset_label_original)
    # train_dataset = []
    # for key1, key2 in zip(train_dataset_original.keys(), train_dataset_label_original.keys()):
    #     train_dataset.append(
    #         InputExample(guid=key1, text_a=train_dataset_original[key1], label=int(train_dataset_label_original[key2])))
    # valid_dataset_original = json.load(open("dataset_own/meld_dev.json"))
    # valid_dataset_label_original = json.load(open("dataset_label/meld_dev.json"))
    # valid_dataset_original = jsonKeys2int(valid_dataset_original)
    # valid_dataset_label_original = jsonKeys2int(valid_dataset_label_original)
    # valid_dataset = []
    # for key1, key2 in zip(valid_dataset_original.keys(), valid_dataset_label_original.keys()):
    #     valid_dataset.append(
    #         InputExample(guid=key1, text_a=valid_dataset_original[key1], label=int(valid_dataset_label_original[key2])))
    #
    # test_dataset_original = json.load(open("dataset_own/meld_test.json"))
    # test_dataset_label_original = json.load(open("dataset_label/meld_test.json"))
    # test_dataset_original = jsonKeys2int(test_dataset_original)
    # test_dataset_label_original = jsonKeys2int(test_dataset_label_original)
    # test_dataset = []
    # for key1, key2 in zip(test_dataset_original.keys(), test_dataset_label_original.keys()):
    #     test_dataset.append(
    #         InputExample(guid=key1, text_a=test_dataset_original[key1], label=int(test_dataset_label_original[key2])))

    # main
    if config.learning_setting == 'full':
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            resume = args.resume,
            test = args.test,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            test_dataset = test_dataset,
        )
        a = 1
    elif config.learning_setting == 'few_shot':
        if config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label = config.sampling_from_train.num_examples_per_label,
                    also_sample_dev = config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev = config.sampling_from_train.num_examples_per_label_dev
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset = train_dataset,
                    valid_dataset = valid_dataset,
                    seed = seed
                )
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    resume = args.resume,
                    test = args.test,
                    train_dataset = train_sampled_dataset,
                    valid_dataset = valid_sampled_dataset,
                    test_dataset = test_dataset,
                )
            else:
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    test = args.test,
                    test_dataset = test_dataset,
                )
            res += result
        res /= len(seeds)
    elif config.learning_setting == 'zero_shot':
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            zero = True,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            test_dataset = test_dataset,
        )

def trainer(EXP_PATH, config, Processor, train_dataset = None, valid_dataset = None, test_dataset = None, resume = None, test = None, zero = False):
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    config.logging.path = EXP_PATH
    # set seed
    set_seed(config.reproduce.seed)

    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config, plm_wrapper_class = load_plm_from_config(config)



    # define template and verbalizer
    if config.task == "classification":
        # define prompt
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
        # load prompt’s pipeline model
        prompt_model = PromptForClassification(plm_model, template, verbalizer, freeze_plm = config.plm.optimize.freeze_para)

    elif config.task == "generation":
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        prompt_model = PromptForGeneration(plm_model, template, freeze_plm = config.plm.optimize.freeze_para, gen_config = config.generation)
    else:
        raise NotImplementedError(f"config.task {config.task} is not implemented yet. Only classification and generation are supported.")

    # process data and get data_loader
    train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, plm_wrapper_class, config, "train") if train_dataset else None
    valid_dataloader = build_dataloader(valid_dataset, template, plm_tokenizer, plm_wrapper_class, config, "dev") if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, template, plm_tokenizer, plm_wrapper_class, config, "test") if test_dataset else None

    if config.task == "classification":
        if config.classification.auto_t or config.classification.auto_v:
            runner = LMBFFClassificationRunner(train_dataset = train_dataset,
                                                valid_dataset = valid_dataset,
                                                test_dataset = test_dataset,
                                                template=template,
                                                verbalizer=verbalizer,
                                                config = config
                                                )
        elif config.verbalizer == "proto_verbalizer":
            runner = ProtoVerbClassificationRunner(model = prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    id2label = Processor.id2label,
                                    config = config
            )
        else:
            runner = ClassificationRunner(model = prompt_model,
                                    train_dataloader = train_dataloader,
                                    valid_dataloader = valid_dataloader,
                                    test_dataloader = test_dataloader,
                                    id2label = Processor.id2label,
                                    config = config
            )
    elif config.task == "generation":
        runner = GenerationRunner(
            model = prompt_model,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            test_dataloader = test_dataloader,
            config = config
        )

    if zero:
        res = runner.test()
    elif test:
        res = runner.test(ckpt = 'best')
    elif resume:
        res = runner.run(ckpt = 'last')
    else:
        res = runner.run()
    return res


if __name__ == "__main__":
    main()