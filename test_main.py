from main import *

def test_imports():
        imports = open('requirements.txt','r').readlines()
        for import_ in imports
            print(eval(import_ + ".__version__")
        print("test on imports complete")

def test_load_data():
        # Step 1: Load and preprocess datasets
        datasets = load_datasets(DATASET_DIR)
        print("test on datasets complete")

def test_finetune():
        # Step 2: Fine-tune models on each dataset
        fine_tune_models(datasets, MODEL_DIR, only=["Center_Political_sample.txt"])
        print("test on mini-finetune complete")

def test_run():
        # Step 3: Example of using the saved models later
        test_input = "You should sell your stocks after [MASK]"
        model_name = "Center_Political_sample.txt_model"  # Replace with the actual dataset filename used
        predicted_text = load_and_use_model(MODEL_DIR, model_name, test_input)
        assert predicted_text.find("crashes") != -1
