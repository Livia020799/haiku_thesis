# haiku_thesis

Nomi dell'algoritmo : KamiHaiku, IssaGen, BashoBot, HaikuSensei

### 1. ***Dataset*** <br> 
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

- [Matsuo Basho haiku dataset](https://github.com/Mig29x/Haiku-Kanjis)., tagged with the pronunciation, date and season.<br>
Number of haiku: 1020.

- There is also a variety of synthetic datasets, like [this one](https://huggingface.co/datasets/davanstrien/haiku_dpo) plus the code used to generate [it](https://github.com/davanstrien/haiku-dpo?tab=readme-ov-file), but they are in english (could a similar thing be done with haiku in japanese?)

- Kinda a [kigo dictionary](https://github.com/Livia020799/Japanese-haiku-kigo-corpus)<br>
Scraped from a [website of haiku](https://ouchidehaiku.com/spring/contents/seasonwords) both in a full comprehensive corpus and divided by seasons.<br>
I keigo sono parole e/o frasi brevi legate alle stagioni ma che non compongono un haiku intero. Possono essere usate poi per la tokenizzazione perché aiutano a determinare l'haiku a quale stagione si riferisce. La repository contiene anche il codice per lo scraping.


### 2. ***Tokenizzazione giapponese*** <br>
[MeCab](https://pypi.org/project/mecab-python3/)(with IPAdic or UniDic dictionary - reccomended for poetry with fugashi for easy Python integration) + kigo dictionary (季語) or seasonal-word database.<br>
Example of MeCab + UniDic and fugashi.<br>
`pip install fugashi[unidic-lite]` *#automatically downloads and sets up UniDic behind the scenes* <br> 
`from fugashi import Tagger`<br>
`# Using fugashi with UniDic`<br>
`tagger = fugashi.Tagger('-Owakati')` *#output tokens separated by spaces* <br>
`haiku_text = "古池や蛙飛び込む水の音"`<br>
`tokenized = tagger.parse(haiku_text).strip().split()`<br>
`print(tokenized)`<br>
[in the study *Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)* they use MeCab]

   
### 3. ***Fine-tuning prompting*** <br>
- [Sakana](https://sakana.ai/series-a/) -> azienda che ha lavorato sulla generazione degli haiku allenando facendo fine tuning/prompting
- Deep Haiku – Teaching GPT-J to Compose with Syllable Patterns: This project involved fine-tuning the GPT-J language model to generate haiku following the traditional 5-7-5 syllabic pattern.<br>
[GitHub repository](https://github.com/robgon-art/DeepHaiku) ***done with an haiku dataset in english***<br>
- *"Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)* -> involved fine-tuning high-performance language models such as GPT-2 and BART

  
### 4. ***Metriche*** <br>
Nella parte di valutazione quali metriche usare (costruire metriche sulla composizine metrica? Chiedere ai GPT come gli sembra essere in termini di aderenza alla struttura metrica)<br>
In the study *"Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)* they use:<br>
1)Perplexity<br>
2)Seasonal Keyword Accuracy<br>
3)Human evaluation<br>
**METRICA AD HOC??**<br>
Chiedere ai GPT come gli sembra essere in termini di aderenza alla struttura metrica


### 5. ***Questionari*** <br>
POssible point to cover for questionnaire:<br>
1)Structure (mora count)
2)Linguistic coherence
3)Poetic imagery
4)Emotional impact
5)Originality and literary aesthetics<br>
+ Tips from the last section of *"Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)*

