import argparse
from tkinter.messagebox import NO
from rouge_score import Rouge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scorer")
    parser.add_argument("--data_path", type=str, default='/home/joonwon/subtitle_generation/extraction_generation/data/dataset/over512/infered_finished_dataset/fullbody_front_1024_truncated_infer_done_with_bert_score.csv')
    parser.add_argument("--reference_column", type=str, help='original summary column name', default='subtitle')
    parser.add_argument("--system_column", type=str, help='generate summary column name', default='generated_subtitle')
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--rouge_list", nargs="+", default=['rouge1','rouge2','rouge3','rougeL'])
    args = parser.parse_args()

    if args.tokenizer_path is not None:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    else : 
        tokenizer = None

    rouge_results = Rouge(data=args.data_path, rouge_types=args.rouge_list, reference_column=args.reference_column, system_column=args.system_column, tokenizer=tokenizer)
    results = rouge_results.rouge_scores()
    for rouge_n in results.keys():
        print(f"{rouge_n} : {results[rouge_n]}")
    
    '''
    *** example ****
    rouge1 : {'recall': 0.23705050797942165, 'precision': 0.2200322066270304, 'f1_score': 0.2171191741634712}
    rouge2 : {'recall': 0.12645253978028817, 'precision': 0.11583582090271122, 'f1_score': 0.11455325679545073}
    rouge3 : {'recall': 0.08118010347329194, 'precision': 0.07342721759642971, 'f1_score': 0.0727209501525823}
    rougeL : {'recall': 0.2142089816202758, 'precision': 0.19684350318385715, 'f1_score': 0.19524798337987434}
    '''