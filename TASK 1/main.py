from agentic_evaluator import evaluate_papers
import pandas as pd
import os
from extractor import process_papers

if __name__ == "__main__":

    df = pd.read_csv('paper_analysis_results.csv')
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total papers processed: {len(df)}")
    print(f"Research categories found: {df['Category of Research'].unique()}")
        

    results_df_test = evaluate_papers(df)
    results_df_test.to_csv('analysis_df.csv', index=False)
    print(results_df_test)