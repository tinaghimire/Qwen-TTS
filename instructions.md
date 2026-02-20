
I have these data 


from datasets import load_dataset

ds = load_dataset("vaghawan/hausa-tts-22k")

and split train validation test

Make use of this directly

and the reference audio is 
Qwen3-TTS-finetuning/voices/english_voice/english_voice.wav


add this to the dataset after loading

use dataloader or other

and then train with more flexibility

new file for dataset


@Qwen3-TTS-finetuning/Qwen3-TTS/finetuning/sft_12hz.py:1-162 

create new train file with Trainer class

and also evaluation

use the loss also for validation

Include wandb

and upload the best and last models to the huggingface account vaghawan/tts-best and vaghawan/tts-last

I need all the checkpoints for last model to continue the finetuning from this including optimizer and schedulers

and for best model can upload the required models needed for inference