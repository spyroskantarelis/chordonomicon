import subprocess

cmd = [
    "python", "Model_Training.py",
    "--representation", "triad",
    "--model_type",     "lstm",
    "--data_path",      "./data/segmented_3tokens.pkl",
    "--vocabs_path",    "./data/vocabs/vocabs.pkl",
    "--embed_dim",      "16",
    "--hidden_dim",     "256",
    "--num_layers",     "1",
    "--batch_size",     "4096",
    "--epochs",         "200",
    "--lr",             "5e-3",
    "--sample_size",    "200000",
    "--num",            "0",
]

cmd2 = [
    "python", "Model_Evaluation.py",
    "--representation", "triad",
    "--model_type",     "lstm",
    "--model_path",          "./models/best_LSTM_model.pth",
    "--model_name",          "LSTM_triad_baseline",
    "--dataset_path",        "./data/segmented_3tokens.pkl",
    "--vocabs_path",         "./data/vocabs/vocabs.pkl",
    "--batch_size",          "4096",
    "--top_n",               "10",
    "--second_dataset_path", "./data/labeled_3tokens.pkl",
    #"--full_dataset",
]

subprocess.run(cmd2, check=True)
