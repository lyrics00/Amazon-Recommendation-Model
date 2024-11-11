from datasets import load_dataset

# Load the Electronics reviews dataset in streaming mode
def load_amazon_dataset():
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", split="full", streaming=True, trust_remote_code=True)

    # Collect only the first 100 entries
    sample_data = []
    for i, entry in enumerate(dataset):
        sample_data.append(entry)
        if i >= 5000:  # Stop after 100 entries
            break

    # Display the first few entries of the sample data
    print(sample_data[:5])
    return sample_data
