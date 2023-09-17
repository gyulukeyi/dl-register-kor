# Can Korean Language Models Detect Social Registers in Utterances?

This is a repository for the routines for the experiments supporting:

> Lee & Song. (to appear). Can Korean Language Models Detect Social Registers in Utterances? _Korean Journal of Linguistics_ (under revision).

## What does it do?

- `utils/handle_data.py`: make unified data file ready for the deep learning objectives. 
    - function `prepare`: gets json data files from National Institute of Korean Language.
    - function `label`: from data converted with `prepare`, make labels using the schemes given as an external `json` file. 
        - file `labeling_scheme.json`: exemplar labeling schema file, as utilized for the published experiments as well.
    - usage: see function `run_prepare` of `main.py`
    - **NOTE** due to the license issue, we do not include the data in any format. For the data, visit `corpus.korean.go.kr`. `prepare` function is designed for the particular format of the data provided here. Particular data we used are titled:
        - 일상 대화 말뭉치 2021 
        - 일상 대화 말뭉치 2020
- `train.py`: perform fine-tuning with PyTorch framework and HuggingFace libraries.
    - Given the data prepared with labels as a `json` file, i.e., as produced by function `run_prepare` in `main.py`, 
        - make PyTorch dataset instances 
        - (`run` and `test` functions will save the dataset into pickles in `datajar` directory for later references by default)
        - load model checkpoint from HuggingFace Hub with sequence classification objective 
        - perform fine-tuning with HuggingFace Trainer with `wandb` report 
- `test_openai.ipynb`: in-context inference test with OpenAI API 
    - Notebook with outputs for:
        - loading test data from local pickle file 
        - constructing prompts 
        - make API calls 
        - write `json` file for RegEx-based cleaning (Expressions were included as a text in `sed`-ready format, not as a code with Python `re`. Since OpenAI output can literally be in any format, it is best to perform the editing on the text file monitoring substitution results with some manual edits. Expressions given were used in Visual Studio Code to make cleaned output file.)
    - **NOTE** you need to provide your own OpenAI API key. By default, the notebook will load the key stored as a plain text file name `.creds`. You can either fill up `.creds` file, or adjust the notebook cells to assign API key with a string literal, or load the key with other methods. 
    - **NOTE** this notebook will make few API calls on your account. Make sure you have your credit ready to make calls. The free credit provided when signing up was more than enough for this, but your mileage can always vary with OpenAI :/
    - **NOTE** OpenAI API calls fail for several reasons from lack of credits to busy server. Unfortunately, error code from the API are way less than enough to make smart adjustments for the failures. As many of such errors are temporal with an instructions to 'try again later', each call is in an infinite loop that make the same call after 10 seconds upon each failure. If it fails too much, stop the cell (send `ctrl+c`) to get out of the loop.