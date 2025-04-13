# haiku_thesis

User study website : [prolific](https://www.prolific.com/)

Overview of japanese tokenizers : [link](https://www.dampfkraft.com/nlp/japanese-tokenizer-dictionaries.html)

[SakanaAI](https://sakana.ai/ai-scientist/) chatbot [Karamaru](https://x.com/SakanaAILabs/status/1906968656120254696) that converses in Edo period language <br> 
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
- *"Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)* involved fine-tuning high-performance language models such as GPT-2 and BART.

  
### 4. ***Metriche*** <br>
Nella parte di valutazione quali metriche usare (costruire metriche sulla composizine metrica? Chiedere ai GPT come gli sembra essere in termini di aderenza alla struttura metrica)<br><br>
***In the study *"Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)**** they use:<br>
1)Perplexity<br>
2)Seasonal Keyword Accuracy<br>
3)Human evaluation<br><br>
***In the study *Quality Estimation for Japanese Haiku Poems  
Using Neural Network - 2016 - (Kikuchi et al.)**** the authors propose a method to estimate the artistic quality of Japanese haiku poems using a machine learning approach. They constructed two distinct vector representations: one word-based, capturing semantic features, and another syllable-based, emphasizing phonetic characteristics. Their evaluation relied on an objective metric derived from user engagement (number of "likes") collected from a large online haiku community dataset. Haiku poems receiving likes above the average were categorized as high-quality, while those below the average were labeled as low quality. Subsequently, convolutional neural networks (CNNs) were trained separately on these vector models to classify the poems. The final quality estimation combined the predictions from both models, achieving improved accuracy.<br><br>
***In the study *Aesthetic evaluation of AI-generated vs. with human intervention - 2023 - (Hitsuwari et al.)**** the authors conducted human-based aesthetic evaluations comparing three types of haiku: (1) human-composed haiku, (2) AI-generated haiku without human intervention (HOTL - Human-Out-of-the-Loop), and (3) AI-generated haiku refined by human selection (HITL - Human-In-the-Loop). To assess the perceived artistic quality, they utilized 21 aesthetic and psychological dimensions, including beauty, valence, empathy, nostalgia, vividness, novelty, and storytelling, among others. Each haiku was rated by 385 participants using a 7-point Likert scale, and statistical analyses such as ANOVA and Linear Mixed Models were applied to determine the key factors influencing the perception of beauty. Additionally, the study included a discrimination task in which participants attempted to distinguish between human and AI-generated haiku, revealing insights into algorithm aversion and perceived creativity in AI-generated literature.<br><br>
Chiedere ai GPT come gli sembra essere in termini di aderenza alla struttura metrica


### 5. ***Questionari*** <br>
Possible point to cover for questionnaire:<br>
1)Structure (mora count)<br>
2)Linguistic coherence<br>
3)Poetic imagery<br>
4)Emotional impact<br>
5)Originality and literary aesthetics<br>
+ Tips from the last section of *"Implementation of Autoregressive Language Models for Generation of Seasonal Fixed-form Haiku in Japanese" (Hirata et al.)*

