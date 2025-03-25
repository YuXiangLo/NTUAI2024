# NTUAI2024

## Task 1

### modelcard
* [Blip](https://huggingface.co/Salesforce/blip-image-captioning-base)
* [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

### dataset
* [mscoco](https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval)
* [flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)

* There are four python files that can generate separate json files which store the captions.
    1. `mscoco\_blip.py`
    1. `mscoco\_phi4.py`
    1. `flickr\_blip.py`
    1. `flickr\_phi4.py`
* The evaluations functions in `(mscoco/flickr)\_(blip/phi4).py` are not from `evaluattions` packages, therefore I wrote a `update\_metric\_result.py` to recalculate.

## Task 2

### modelcard
* **_MLLM Model:_** [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
* **_T2I Model:_** [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
* **_I2I Model:_** [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

### dataset
* [CeleFace](https://drive.google.com/file/d/1VU3yMVG_MyDUTBkRIZxmIu1-tUkHzuJT/view)

* `get_captions.py` uses phi-4 to caption images.
* `diff(3/1\_5).py` uses diffusion model to generate styled images.

### Restriction
* Output Image Format: $224 \times 224$
* No additional model or fine tuning
