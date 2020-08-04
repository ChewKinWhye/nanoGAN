from data2 import load_data
from arguments import parse_args

args = parse_args
x_test, y_test = load_data(args)

predictions = []

# Load model

with open(args.data_path, 'r') as f
    for row in test_x:
        row_input = []
        for index in row[1:]:
            temp_data = next(itertools.islice(csv.reader(f), index, None))
            # Normalize and stuff
            row_input.append(temp)
        
        temp_prediction = 1 - np.squeeze(model.predict_on_batch(row_input))
        predictions.append(sum(temp_prediction)/len(temp_prediction))

print(len(predictions))
best_accuracy, best_sensitivity, best_specificity, best_precision, best_au_roc, best_cm = compute_metrics_standardized(predictions, y_test)
