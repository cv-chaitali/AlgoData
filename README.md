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