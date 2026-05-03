import re
from typing import List, Dict

def parse_population_file(file_path: str) -> List[Dict]:
    """
    Parse population file and extract individual information.

    Args:
        file_path: Path to the population file

    Returns:
        List of dictionaries containing 'indi', 'code', and 'acc' for each individual
    """
    results = []

    with open(file_path, 'r') as f:
        content = f.read()

    # Split by the separator line
    blocks = content.strip().split('-' * 100)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract indi
        indi_match = re.search(r'indi:(indi\d+)', block)
        # Extract code
        code_match = re.search(r'code:(.+)', block)
        # Extract accuracy
        acc_match = re.search(r'Acc:(\d+\.\d+)', block)

        if indi_match and code_match and acc_match:
            indi = indi_match.group(1)
            code_str = code_match.group(1).strip()
            acc = float(acc_match.group(1))

            # Parse code into list of integers
            code_values = [int(x.strip()) for x in code_str.split(',')]

            # Restructure code as [first, second, [middle 5], [last 5]]
            code = [
                code_values[0],           # first element
                code_values[1],           # second element
                code_values[2:7],         # middle 5 elements
                code_values[7:]           # last 5 elements
            ]

            results.append({
                'indi': indi,
                'code': code,
                'acc': acc
            })

    return results

def parse_population_file_to_dataframe(file_path: str):
    """
    Parse population file and return as pandas DataFrame.
    """
    import pandas as pd

    results = parse_population_file(file_path)

    # Expand code into separate columns
    data = []
    for item in results:
        row = {
            'indi': item['indi'],
            'acc': item['acc']
        }
        # Add each code element as a separate column
        for i, code_val in enumerate(item['code']):
            row[f'code_{i}'] = code_val
        data.append(row)

    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    file_path = "results/training_based/pop10_gen20/seed41/populations_PT_seed41_ewc/populations/begin_19.txt"

    # Parse as list of dictionaries
    results = parse_population_file(file_path)

    print(f"Found {len(results)} individuals")
    print("/nAll results:")
    for i, result in enumerate(results):
        print(f"/n{i+1}. {result['indi']}")
        print(f"   Code: {result['code']}")
        print(f"   Accuracy: {result['acc']:.3f}")

    # Parse as DataFrame (optional)
    df = parse_population_file_to_dataframe(file_path)
    print("/n\nDataFrame shape:", df.shape)
    print("\nDataFrame head:")
    print(df.head())

    # Find best individual
    best_indi = max(results, key=lambda x: x['acc'])
    print(f"\n\nBest individual: {best_indi['indi']} with accuracy {best_indi['acc']:.3f}")
    print(f"Code: {best_indi['code']}")
