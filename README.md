# Yandex Cup ML Challenge '23
## RecSys Track

[**Задача**](https://yandex.ru/cup/ml/): Предсказание музыкального жанра по звучанию трека. Каждый трек описывается последовательностью ембедингов размерности 768, посчитанных по фрагменту аудио трека фиксированной длины.

**Public Avg PR**: *0.3067* \
**Private Avg PR**: *0.3114*

To reproduce the results:
1. `conda create -n ya python=3.10`, `source activate ya`, `pip install -r requirements.txt`
2. Run scripts: `scripts/run_train_models_0.sh`,`scripts/run_train_models_1.sh`,`scripts/run_train_models_2.sh`,`scripts/run_train_models_3.sh`
3. Average results using  `average.ipynb`

**Leaderboard:** 

<img width="609" alt="image" src="https://github.com/ArtemVazh/ya_cup23/assets/44724467/f87da92c-3371-4c01-8430-b0573a247393">





