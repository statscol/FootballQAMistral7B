# Football Rag Chat

Using FIFA data to get insights from football matches

Data source: [Kaggle](https://www.kaggle.com/datasets/zeesolver/fifa-results?resource=download)

The approach for this demo consist of creating a SQLliteDB from the csv files which will be fed to the LLM and the SQL agent using Langchain, the LLM selected is Mistral 7B-InstructV0.3. (See Figure below)

![img](./data/description.png)


## 🪒 Setup

Using conda or virtualenv install the packages.

```bash
conda create --name <YOUR_ENV_NAME> python=3.10
conda activate <YOUR_ENV_NAME>
pip install -r requirements.txt
```

Make sure you have at least 12 GB of VRAM.

## 🐍 Usage



After installing the packages in a venv or a Conda env, you can run the following to test the pipeline


- Creating DB
    ```bash
    python utils/create_db.py
    ```

    This will create a .db file in the `data` folder. An example is also uploaded. 

- Running Inference

    ```bash
    python utils/inference.py
    ```
    You can modify the input question and play around with this script

### Demo

Using the default question in `utils/inference.py` this is the expected output

![img](./data/demo.png)

Also a gradio App will be available (WIP)

## 🤿 Contributing to this repo

- This repo uses pre-commit hooks for code formatting and structure. Before adding|commiting changes, make sure you've installed the pre-commit hook running `pre-commit install` in a terminal. After that changes must be submitted as usual (`git add <FILE_CHANGED> -> git commit -m "" -> git push `)

- For dependencies, [pip-tools](https://github.com/jazzband/pip-tools) is used. Add the latest dependency to the requirements.txt and run  `pip-compile requirements.txt -o requirements.txt` to make sure the requirements file is updated so we can re-install with no package version issues.
