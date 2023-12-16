# 4th Place Solution (monnu's part)

## Environment
Use the following commands to set up the environment:
```
docker-compose up -d --build
docker exec -it kaggle bash
```

## Usage
0. **Data Preparation**: Place competition data in the `input` directory.

1. **Base Pair Matrix (EternaFold)**: Generate a base pair matrix using EternaFold. For this step, use scripts provided by my teammate [tattaka](https://github.com/tattaka/stanford-ribonanza-rna-folding-public?tab=readme-ov-file#usage).

2. **Training**: Run the following scripts to train the models.
    ```bash
    cd src/exp112 && sh train.sh
    cd src/exp300 && sh train.sh
    cd src/exp302 && sh train.sh
    cd src/exp312 && sh train.sh
    cd src/exp317 && sh train.sh
    ```

3. **Ensemble**: Generate out-of-fold (oof) predictions and submissions using all 12 models from our team. Then, execute the ensemble script:
    ```bash
    python src/scripts/ensemble.py
    ```

## License
MIT