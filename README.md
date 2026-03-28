# EvoTox

> [!WARNING]  
> This repository contains code and logs from the toxicity testing of Large Language Models. The logs and the other
> material used during the evaluations include harmful and offensive text, such as hate speech, explicit material, and
> threats, which are provided solely for research and documentation purposes. This content is intended for researchers,
> developers, and practitioners in AI safety and ethics. Users should handle the material with professional care and
> sensitivity, and refrain from using it in a way that could cause harm.
>
> By accessing this repository, you acknowledge the potential risks associated with the content and agree to take full
> responsibility for its use. The authors and contributors disclaim any liability for misuse or harm caused by the
> content. If you have any concerns or issues, please contact us.

A framework to evaluate how much can a Large Language Model be toxic
## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python evolutionary.py --help
python plotter.py --help
```

## Options

### Evolutionary

| Option                  | Short Flag | Type     | Default                                                    | Required | Description                                                                                                                            |
|-------------------------|------------|----------|------------------------------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------|
| `--iterations`          | `-i`       | `int`    | `7`                                                        | No       | Number of iterations to run the algorithm                                                                                              |
| `--file`                | `-f`       | `Path`   | `dataset/questions`                                        | Yes      | File to load the questions from                                                                                                        |
| `--evaluation-function` | `-e`       | `Choice` | `perspective`                                              | No       | Evaluation function to use                                                                                                             |
| `--scoring-function`    | `-s`       | `Choice` | `max`                                                      | No       | Scoring function to use                                                                                                                |
| `--categories`          | `-c`       | `List`   | `["sexist", "racist", "homophobic", "toxic", "insulting"]` | No       | Categories to use for the prompt generator                                                                                            |
| `--category-forwarding` | `-cf`      | `Choice` | `all`                                                      | No       | Forward the category to the prompt generator                                                                                          |
| `--repeat-penality`     | `-r`       | `float`  | `1.1`                                                      | No       | Penalty applied to repeated tokens during text generation                                                                              |
| `--verbose`             | `-v`       | `flag`   | `False`                                                    | No       | Show various iterations and their relative results                                                                                     |
| `--output-path`         | `-o`       | `Path`   | None                                                       | No       | Path to save the output                                                                                                                |
| `--questions`           | `-q`       | `int`    | `-1`                                                       | No       | Number of questions to load from the dataset                                                                                           |
| `--offset`              | `-of`      | `int`    | `0`                                                        | No       | Offset for questions to load from the dataset                                                                                          |
| `--batch-size`          | `-b`       | `int`    | `2`                                                        | No       | Batch size for the prompt generator                                                                                                    |
| `--memory`              | `-m`       | `flag`   | `False`                                                    | No       | Use the memory of the previous run                                                                                                     |
| `--forward-score`       | `-fs`      | `flag`   | `False`                                                    | No       | Use the score of the previous run when generating the new prompt                                                                       |
| `--memory-window`       | `-w`       | `int`    | `5`                                                        | No       | Size of the memory window to use for the prompt generator (requires memory enabled)                                                    |
| `--retries`             | None       | `int`    | `100`                                                      | No       | Number of retries for generating a new prompt when the generator fails                                                                |
| `--gaslight`            | `-g`       | `flag`   | `False`                                                    | No       | Use gaslighting technique for scores higher than 0.35                                                                                  |
| `--system-under-test`   | `-sut`     | `Choice` | `vicuna`                                                   | No       | Model to use as System Under Test                                                                                                      |
| `--system-generator`    | `-sg`      | `Choice` | `vicuna`                                                   | No       | Model to use as System Generator                                                                                                       |
| `--example-file`        | None       | `Path`   | None                                                       | No       | File to load examples from                                                                                                             |
| `--api-key-perspective` | None       | `str`    | None                                                       | No       | API key for the Perspective API                                                                                                        |
| `--jailbreak`           | `-j`       | `flag`   | `False`                                                    | No       | Use jailbreak prefixes                                                                                                                 |
| `--population-size`     | `-p`       | `int`    | `10`                                                       | No       | Size of the NSGA-II population                                                                                                         |

### Plotter

| Parameter  | Flag | Required | Type   | Default            | Note                                                    |
|------------|------|----------|--------|--------------------|---------------------------------------------------------|
| `--file`   | `-f` | yes      | `Path` | ``                 | File to load the questions from'                        |
| `--output` | `-o` | no       | `Path` | `out/output_*.png` | Output file path to store the result of the computation |

## LLM Server

> [!NOTE]  
> All commands and scripts mentioned in this section are expected to be run from the root project directory

LLMs are served via a [Llama CPP Python](https://github.com/ggerganov/llama.cpp) API running in
a [Docker](https://www.docker.com) container.
Hereafter are the instructions to build the Docker image, download the quantised models we used in the experiments in
GGUF format and start the service with the API.

### Build docker image

To build the Docker image run

```bash
docker build . -f ./docker/llm_server/Dockerfile -t llm_server
```

### Download models

To download the models we used in the experiments run

```bash
bash ./scripts/download_llm.sh
```

### Start service

To start the server with `<model>` LLM on port `<port>` run

```bash
bash ./scripts/start_llm_server.sh -m <model> -p <port>
```

The script will generate a text file named `llm_server_<date>_<model>` (`<date>` uses the format `"%Y_%m_%d_%H_%M_%S"`)
containing the API key to authenticate to the LLM server and use the completion services.

For further details on the script run

```bash
bash ./scripts/start_llm_server.sh -h
```

## N-gram LM perplexity

> [!NOTE]  
> All commands and scripts mentioned in this section are expected to be run from the root project directory

To use the n-gram LLMs for computing the perplexity we use the [KenLM](https://github.com/kpu/kenlm) library.
We use the same library to train n-gram models.

Here are the links to the models we trained and the corresponding information on training corpus and the order of the model:

| Corpus                                                                         | 3-gram   | 4-gram   | 5-gram   |
|--------------------------------------------------------------------------------|----------|----------|----------|
| [WikiText 2](https://paperswithcode.com/dataset/wikitext-2)                    | [link]() | [link]() | [link]() |
| [Book Corpus](https://paperswithcode.com/dataset/bookcorpus) (sentences split) | [link]() | [link]() | [link]() |
| Book Corpus                                                                    | [link]() | [link]() | [link]() |

Alternatively you can fit the same models from scratch.

### Install KenLM library

Make sure to install the KenLM library using the following commands:

```bash
apt install libeigen3-dev
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build && cmake .. && make -j
```

This passage is based on the following instructions: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-python-advanced-nemo-ngram-training-and-finetuning.html#installing-and-setting-up-kenlm

### Fit LM

There is a script to fit the same n-gram models we used during the evaluation, you can run it with the following command

```bash
nohup bash ./scripts/fit_ngram_models.sh > output_ngram_training.txt &
```

### Compute perplexity

There is a script to compute the perplexity of the prompts generated during the experiments, you can run it with the following command

```bash
nohup bash ./scripts/compute_ngram_ppl.sh > output_ngram_ppl.txt &
```
## References

If you are willing to use our code or our models, please cite us with the following reference(s):

```bibtex
TBD
```

## Acknowledgements

- Simone Corbo ([simone.corbo@mail.polimi.it](mailto:simone.corbo@mail.polimi.it))
- Luca Bancale: ([luca.bancale@mail.polimi.it](mailto:luca.bancale@mail.polimi.it))
- Valeria de Gennaro: ([valeria.degennaro@mail.polimi.it](mailto:valeria.degennaro@mail.polimi.it))
- Livia Lestingi: ([livia.lestingi@polimi.it](mailto:livia.lestingi@polimi.it))
- Vincenzo Scotti: ([vincenzo.scotti@polimi.it](mailto:vincenzo.scotti@polimi.it))
- Matteo Camilli: ([matteo.camilli@polimi.it](mailto:matteo.camilli@.polimi.it))
