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

---

## Visualize Umap
|CRAIG|SG Facility|SG Norms|
|---|---|---|
|![craig_selection](https://github.com/user-attachments/assets/b4919f62-b450-4f63-a9b2-4395ed4aebc1)|![sg_facility](https://github.com/user-attachments/assets/845f2f7d-3d91-417a-bd38-30621c5a6f50)|![sg_norms](https://github.com/user-attachments/assets/bcd53a39-dbdf-4d9c-9454-6b9558c61f1f)|

---

## Quantitative Results - 
### Evaluation Details

- **Tasks**:
  - **Wikitext**: evaluated using *Perplexity (PPL)*
  - **ARC-Easy**: evaluated using *Accuracy*
- **Model**:
  - *LLaMA 3.2 1B*
- **Pruning**:
  - *Pruning Ratio*: **0.35**
  - *Pruning Scheme*: **LLM-Pruner**


#### CRAIG Subset Evaluation (Accuracy in %)

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



#### PBC Subset Evaluation (Accuracy in %)

| Data (%) | Perplexity (PPL) | Accuracy (%) |
|----------|------------------|---------------|
| 10       | 34.2700          | 52.74         |
| 20       | 34.7600          | 55.77         |
| 30       | 31.8683          | 55.09         |
| 40       | 30.7200          | 56.10         |
| 50       | 31.1199          | 56.27         |
| 60       | 29.7812          | 54.97         |
| 70       | 29.6246          | 55.72         |
| 80       | 32.1700          | 55.89         |
| 90       | 33.8011          | 55.68         |


#### SG Facility Subset Evaluation (Accuracy in %)

| Data (%) | Perplexity (PPL) | Accuracy (%) |
|----------|------------------|---------------|
| 10       | 51.2900          | 53.79         |
| 20       | 50.9187          | 54.59         |
| 30       | 41.0800          | 54.42         |
| 40       | 31.2409          | 55.64         |
| 50       | 30.2438          | 56.19         |
| 60       | 29.6893          | 56.06         |
| 70       | 29.2839          | 58.00         |
| 80       | 29.2123          | 55.60         |
| 90       | 29.2998          | 57.20         |


#### SG Norms Subset Evaluation (Accuracy in %)

| Data (%) | Perplexity (PPL) | Accuracy (%) |
|----------|------------------|---------------|
| 10       | 50.1700          | 53.58         |
| 20       | 32.4597          | 53.54         |
| 30       | 31.4382          | 54.38         |
| 40       | 33.0700          | 54.25         |
| 50       | 31.1400          | 54.88         |
| 60       | 30.2179          | 56.40         |
| 70       | 29.6715          | 56.06         |
| 80       | 30.5075          | 55.89         |
| 90       | 29.1851          | 55.93         |




