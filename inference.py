import logging
from transformers import LlamaTokenizerFast, LlamaForCausalLM
import os
import sys

from train import PROMPT_DICT, DEFAULT_EOS_TOKEN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

model = LlamaForCausalLM.from_pretrained(
    os.environ.get("TRAINML_CHECKPOINT_PATH"), device_map="auto"
)
tokenizer = LlamaTokenizerFast.from_pretrained(
    os.environ.get("TRAINML_CHECKPOINT_PATH"),
)
model.eval()


def instruct(
    instruction,
    input=None,
    max_tokens=2048,
    temperature=1,
    top_p=0.75,
    num_beams=1,
):
    prompt = (
        PROMPT_DICT["prompt_input"].format(
            instruction=instruction, input=input
        )
        if input
        else PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generate_kwargs = dict(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_length=max_tokens,
        num_beams=num_beams,
    )
    gen_tokens = model.generate(input_ids, **generate_kwargs)
    response_count = gen_tokens.size()[1] - input_ids.size()[1]

    print(f"total_tokens: {gen_tokens.size()[1]}")
    print(f"prompt_tokens: {input_ids.size()[1]}")
    print(f"completion_tokens: {response_count}")
    gen_text = tokenizer.batch_decode([gen_tokens[0][-response_count:]])[0]
    return gen_text.removesuffix("<s>").removesuffix(DEFAULT_EOS_TOKEN)
