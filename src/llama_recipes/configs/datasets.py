# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpacbiza_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class AMR2_dataset:
    dataset: str = "AMR2_dataset"
    file: str = "AMR_function/custom_dataset.py"
    data_path: str = "AMR_function/AMR2.0"
    train_split: str = "train"
    test_split: str = "dev"

@dataclass
class AMR3_dataset:
    dataset: str = "AMR3_dataset"
    file: str = "AMR_function/custom_dataset.py"
    data_path: str = "AMR_function/AMR3.0"
    train_split: str = "train"
    test_split: str = "dev"

@dataclass
class SilverData:
    dataset: str = "SilverData"
    file: str = "AMR_function/custom_dataset.py"
    data_path: str = "AMR_function/SilverData"
    train_split: str = "train"
    test_split: str = "dev"