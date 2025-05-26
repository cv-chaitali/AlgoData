# ALGORITHM 



## Run script
1. Create Conda env
    ```
    conda create -n algo python=3.8 -y
    conda activate algo
    ```

2. Install pakages
    ```
    pip install -r requirements.txt
    ```

3. run script
    ```
    bash select_auto.sh
    ``` 

---

## Automatic upload to hugging face

1. Make Access Token in [Hugging Face Tokens](https://huggingface.co/settings/tokens)

2. CLI login in Terminal
    ```
    huggingface-cli login
    ```
    Enter the access token

3. Run python script
    ```
    python upload_hugging_face.py 
    ```


### Quantitative Results 

## CRAIG Subset Evaluation (ARC-Easy Accuracy in %)

| Data (%) | Perplexity (PPL) | Accuracy (%) |
|----------|------------------|---------------|
| 10       | 33.6824          | 53.37         |
| 20       | 31.9796          | 54.59         |
| 30       | 31.6592          | 53.54         |
| 40       | 31.4817          | 54.34         |
| 50       | 36.2250          | 55.56         |
| 60       | 29.9951          | 55.05         |
| 70       | 29.6320          | 55.64         |
| 80       | 33.6755          | 56.19         |
| 90       | 31.6105          | 55.68         |
