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
