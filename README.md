# Evaluating The Cognitive Plausibility of Transformer-Based Models: Predicting Articulation Rate from Surprisal Estimates

Surprisal is a measure of the amount of information carried by a linguistic unit (e.g., a word), which is inversely related to its predictability: high predictable words have low-information content and unexpected words carry more information. Research has shown a negative relationship between surprisal and speech tempo—such as articulation rate (AR)—as reflected in human behavioral data, including reading times. Transformer-based Large Language Models (LLMs) like GPT-2 produce surprisal estimates that are strongly predictive of reading times in humans, effectively capturing this negative relationship. However, recent analyses have observed a positiverelationship between the Psychometric Predictive Power (PPP) of larger GPT-2 models and reading times—compared to smaller GPT-2 variants—suggesting that larger models poorly approximate human reading behavior. However, it remains unexplored whether surprisal estimates from BERT can better predict AR in read speech than GPT-2. The aim of this work is to evaluate which of GPT-2 and BERT are better cognitive models of language processing in humans. We employ General Linear Models (GLMs) to model AR as a function of surprisal estimates from both models, while controlling for word position, word length, and final lengthening. The results suggest that BERT is a more cognitively plausible model of human language processing than GPT-2, based on two observations. Firstly, BERT models trained with smaller context window sizes obtained surprisal estimates that led to improved PPP. Secondly, the smallest BERT model—based on a context window size of 128 tokens–achieved the lowest Akaike Information Criterion (AIC) score, suggesting a better fit to the observed data. We argue that our result have broader implications for cognitive modeling, especially the possibility of BERT to capture different psycholinguistic processes in humans than GPT-2—language planning, retrospective simulation, and the interaction between prediction and comprehension. Our results question prior assumptions that BERT is a cognitively implausible model. 

## 1 Pipeline
The following pipeline was built to pre-process speech recordings of the King James Version Bible from the [Faith Comes by Hearing](https://www.faithcomesbyhearing.com/) database. The chosen collection corresponds to the ID-code ENGKJV, from which 28 New Testament books are contained. The entire collection needs to be downloaded manually from the recording [database](https://www.faithcomesbyhearing.com/audio-bible-resources/mp3-downloads). Create a root folder `ENGKJV` with the subfolders `audio` for storing the 260 audio recorings in mp3, `text` for storing the orthographies, and `text_grids` for storing the outputs of the forced aligner.

### 1.1 Audio segmentation

The script `audio_segment_pipe.py` was used to trim the introductory and concluding speeches across the 260 audio files. Each file contains an introductory speech and concluding speeches are found in the last audio of each bookw. The principle is to trim the last pause before the real speech starts and the first pause before the concluding speech ends at the midpoint. Note that different languages do not necessarily contain these extra speeches, and the extent of both the introductory and concluding speeches varies. Then, all the 260 audio recordings are converted to `.wav` using the script `convert_to_wav.py`.

### 1.2 Text and Audio Pre-processing
The 260 Bible texts are already provided as JSON files in the `json` folder for practical reasons. The `OrthographicText` class in the function `orth_class_pipe.py` with the flag `open_jason` set to `True` will open the saved files without the need to re-run the web scraping and parsing process with the `BeautifulSoup` library. The function `_normalize` normalizes the Bible texts by lowercasing and stripping punctuation. The 260 normalized texts are then saved in the `text` folder for further conversion into phonetic transcriptions.

The `ExtractFeatures` class in the `feature_class_pipe.py` scripts inherits the functionalities of the `OrthographicText` class. This class will extract articulation rates, estimate word lengths (i.e., the total number of syllables), and assign the values `0` and `1` to the dummy variables `first` and `last` (i.e., to the first and last word within each sentence). The flag `statistics` set to `True` prints useful information on the screen for debugging, such as the total token count for each chapter. Running the script with the flag `write_to_file` set to `True` will save five CSV tables in the folder. The CSV table based on the `ALL_DATA` dataframe contains all the variables.

The Grapheme-to-Phoneme ([G2P](https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/Grapheme2Phoneme)) web service provided by [WebMAUS](https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface) was employed to convert the orthographic texts into phonetic transcriptions in SAMPA. The script  `extract_par_files.py` runs the service remotely, saving 260 `.par` files in the same `audio` folder. 

### 1.3 Alignment of Text to Audio

The MAUS [wrapper](https://www.bas.uni-muenchen.de/forschung/Bas/maus.web) is used to aligned the orthographic texts and transcriptions to the trimmed audio files at the word level by remotely calling the web service. The script `wrapper_pipe.py` implements the wrapper parameters in the `maus_wrapper` function. The results are 260 TextGrids containing three levels of annotations: orthographic text, SAMPA transcriptions, and phoneme-level annotation, saved to a folder. Note that this process can take between 1-2 hours.

## 2. Training

The scripts `train-gpt2.py` and `train-bert.py` implement the pre-trained GPT-2 and BERT models (124M and 110M parameters) and several classes of the `transformers` library from [HugginFace](https://huggingface.co/docs/transformers/en/index). The models are trained at [UPPMAX](https://www.uu.se/centrum/uppmax/), Uppsala University, using a single GPU.

The training data is the Wiki40b dataset from HuggingFace, constisting of English Wikipedia data. The data was saved as `.parquet` files in a specific folder at [UPPMAX](https://www.uu.se/centrum/uppmax/), but the code can be implemented with the `load_dataset` function (i.e., `load_dataset("wiki40b", "en")`). 

The variable `rows` contains the estimated rows for reaching the upper bound of 90M sub-tokens for each context window size. 

The training functions are integrated in six Bash scripts that run the following command:

- `python3 train-gpt2.py 512`

where `512` is the current context window size used during training. After running, the best models are saved in separate folders.


## 3. Surprisal Estimation

Surprisal estimation is conducted by the `eval_surp.py` function in the `extract_surprisal_gpt2.py` and `extract_surprisal.py` scripts. The function `merge_subtokens` reconstructs the words by merging the sub-tokens and summing up their surprisal estimates. Surprisal estimation is based on the chapter level rather than sentence level, to ensure more context. The `filter_subtokens` function is an extra post-processing step that filters unprocessed sub-tokens in the merging procedure. Each script can be run with the following command:

- `python3 extract_surprisal_gpt2.py ENGKJV gpt 512 True`

Change `gpt`to `bert` and `512` to the desired window size. The `write` flag set to `True` will create a CSV file with two columns: the reconstructed tokens and their surprisal estimates. Each CSV are padded with `None` tokens to ensure index alignment across different context windows.

## 4. Statistical Models

The Research Questions (RQ1 and RQ2) are addressed using six Generalized Linear Models (GLMs), using the [`statsmodels`](https://www.statsmodels.org/stable/examples/notebooks/generated/glm.html) library. The results for RQ1 are based on the `rq1_jsd.py` and `rq1_ppp.py` scripts, and the results for RQ2 are based on the `rq2_glms.py` script. 

The log-likelihood and Delta values of the `data` variable in the `rq1_ppp.py` script have been obtained by the `StatisticalModel` class in the `rq2_glms.py` script. Here, the fuction `glmm` creates six full GLMs, while six null baselines GLMs (i.e., without the surprisal predictor and the interaction variables) are created by the `null_model` variable. The results of the `glms` function, consisting of the Akaike's Information Criterion ([AIC](https://link.springer.com/rwe/10.1007/978-3-642-04898-2_110)) score, the log-likelihood, and the transformed coefficients in syllables per second, are printed on the screen, or can be saved with the `write_summary` flag set to `True`.
