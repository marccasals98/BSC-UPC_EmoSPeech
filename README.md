# BSC-UPC at EmoSPeech-IberLEF2024: Attention Pooling for Emotion Recognition
This repository is a public version of the repository that was used in the EmoSPeech challenge 2024. 

## Abstract
The domain of speech emotion recognition (SER) has persistently been a frontier within the landscape of machine
learning. It is an active field that has been revolutionized in the last decades and whose implementations are
remarkable in multiple applications that could affect daily life. Consequently, the Iberian Languages Evaluation
Forum (IberLEF) of 2024 held a competitive challenge to leverage the SER results with a Spanish corpus. This
paper presents the approach followed with the goal of participating in this competition. The main architecture
consists of different pre-trained speech and text models to extract features from both modalities, utilizing an
attention pooling mechanism. The proposed system has achieved the first position in the challenge with an
86.69% in Macro F1-Score

## How to run the code
1. Create a python environment (or Conda):

`python -m venv your_environment`

`source your_environment/bin/activate`

2. Install requirements:

`pip install -r requirements.txt`

3. Run the desired training with SLURM:

`sbatch shs/3_XLSR_ROBERTA-esp.sh`

## References
Chiruzzo, L., Jiménez-Zafra, S. M., & Rangel, F. (2024). Overview of IberLEF 2024: Natural Language Processing Challenges for Spanish and other Iberian Languages. Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2024), Co-Located with the 40th Conference of the Spanish Society for Natural Language Processing (SEPLN 2024), CEUR-WS.Org.

Pan, R., García-Díaz, J. A., Rodríguez-García, M. Á., García-Sánchez, F., & Valencia-García, R. (2024). Overview of EmoSPeech 2024@IberLEF: Multimodal Speech-text Emotion Recognition in Spanish. Procesamiento Del Lenguaje Natural, 73(0).

Pan, R., García-Díaz, J. A., Rodríguez-García, M. Á., & Valencia-García, R. (2024). Spanish MEACorpus 2023: A multimodal speech-text corpus for emotion analysis in Spanish from natural environments. Computer Standards & Interfaces, 103856.

## Citing
[1] M. Casals-Salvador, F. Costa, M. India, and J. Hernando, “BSC-UPC at EmoSPeech-IberLEF2024: Attention Pooling for Emotion Recognition,” Jul. 17, 2024, arXiv: arXiv:2407.12467. doi: 10.48550/arXiv.2407.12467.



```bib
@misc{casals-salvador_bsc-upc_2024,
	title = {{BSC}-{UPC} at {EmoSPeech}-{IberLEF2024}: {Attention} {Pooling} for {Emotion} {Recognition}},
	copyright = {Creative Commons Attribution 4.0 International Licence},
	shorttitle = {{BSC}-{UPC} at {EmoSPeech}-{IberLEF2024}},
	url = {http://arxiv.org/abs/2407.12467},
	doi = {10.48550/arXiv.2407.12467},
	abstract = {The domain of speech emotion recognition (SER) has persistently been a frontier within the landscape of machine learning. It is an active field that has been revolutionized in the last few decades and whose implementations are remarkable in multiple applications that could affect daily life. Consequently, the Iberian Languages Evaluation Forum (IberLEF) of 2024 held a competitive challenge to leverage the SER results with a Spanish corpus. This paper presents the approach followed with the goal of participating in this competition. The main architecture consists of different pre-trained speech and text models to extract features from both modalities, utilizing an attention pooling mechanism. The proposed system has achieved the first position in the challenge with an 86.69\% in Macro F1-Score.},
	urldate = {2024-07-23},
	publisher = {arXiv},
	author = {Casals-Salvador, Marc and Costa, Federico and India, Miquel and Hernando, Javier},
	month = jul,
	year = {2024},
	note = {arXiv:2407.12467 [eess]},
	keywords = {Electrical Engineering and Systems Science - Audio and Speech Processing, Speech Classification},
	file = {arXiv Fulltext PDF:/home/marc/Zotero/storage/ERRS52GJ/Casals-Salvador et al. - 2024 - BSC-UPC at EmoSPeech-IberLEF2024 Attention Pooling for Emotion Recognition.pdf:application/pdf;arXiv.org Snapshot:/home/marc/Zotero/storage/UPSW93US/2407.html:text/html},
}
```

