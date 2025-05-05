from datasets import load_dataset
import pandas as pd


def data_loader():
# Step 1: Load dataset
    dataset = load_dataset("rjac/e-commerce-customer-support-qa", split="train")

    # Step 2: Convert to DataFrame
    df = dataset.to_pandas()
    return df