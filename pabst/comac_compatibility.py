from transformers import (GPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from models.reinforce_model.data import PADDED_INPUTS, ATTR_TO_SPECIAL_TOKEN
from models.reinforce_model.model_with_inferencenw import LatentVariableInferenceModel
import torch
import os 

training_args_path = '/data3/bodhi/projects/persona-dialog/models/reinforce_model/runs/Apr01_15-15-56_deepx_gpt2reinforce0.8_prior_bow_comet/model_training_args.bin'
model_checkpoint_path = '/data3/bodhi/projects/persona-dialog/models/reinforce_model/runs/Apr01_15-15-56_deepx_gpt2reinforce0.8_prior_bow_comet/checkpoint_mymodel_86940.pth'

def load_compac_model(training_args_path, model_checkpoint_path):

    training_args = torch.load(training_args_path)
    model_class = GPT2LMHeadModel
    model = LatentVariableInferenceModel(training_args, generator_class=model_class)

    # for compatibility
    training_args.entropy_regularize_prior_wt = None
    training_args.use_structured_prior = False
    training_args.use_structured_prior_binarypotential = False
    training_args.prior_model = 'roberta'

    # load model class
    model = LatentVariableInferenceModel(training_args, generator_class=model_class)

    # adding extra tokens
    tokenizer_class = GPT2Tokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained('gpt2')
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    model.gpt2_model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

    # model weigths
    model_weights = torch.load( model_checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_weights, strict=False)

    # only take out the generator
    generator = model.gpt2_model

    return generator

