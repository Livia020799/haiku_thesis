# haiku_thesis

Nomi dell'algoritmo : KamiHaiku, IssaGen, BashoBot, HaikuSensei

1. Dataset<br> 
- Haiku Generation Using Deep Neural Networks, 2017, Xianchao Wu.<br> contact : {xiancwu, momokl, kito, zhanc}@microsoft.com <br>
- [Hugging face dataset - jp](https://huggingface.co/datasets/p1atdev/modern_haiku/blob/main/README.md)<br>
Collected from [Modern Haiku Association](https://haiku-data.jp/index.php)<br>
`DatasetDict({` <br>
    `train: Dataset({` <br>
        `features: ['id', 'haiku', 'author', 'foreword', 'source', 'comment', 'reviewer', 'note', 'season', 'kigo'],` <br>
        `num_rows: 37158` <br>
    `})` <br>

`})` <br>
Authors: 4625 (of which approximately 100 are famous haiku poets according to ChatGPT) & other info in **`HuggingFace_haiku.ipynb`**

2. Tokenizzazione giapponese<br>
   
3. Fine-tuning prompting<br>
- Sakana -> azienda che ha lavorato sulla generazione degli haiku allenando facendo fine tuning/prompting
- Deep Haiku – Teaching GPT-J to Compose with Syllable Patterns: This project involved fine-tuning the GPT-J language model to generate haiku following the traditional 5-7-5 syllabic pattern.<br>
[GitHub repository](https://github.com/robgon-art/DeepHaiku) ***done with an haiku dataset in english***
  
4. Metriche<br>
Nella parte di valutazione quali metriche usare (costruire metriche sulla composizine metrica? Chiedere ai GPT come gli sembra essere in termini di aderenza alla struttura metrica)

5. Questionary<br>
Come farli / a chi darli
