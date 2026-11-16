import csv
import math
import os
import os.path as osp
import re
import sys

import torch
from PIL import Image

from .base import BaseModel

# Per-dataset prompts. Two styles:
# - training (default): the prompt the velvet-vision training evaluation gives the model, built from the TSV row
#   itself (``build_training_question``) -- question, ". You also know this: <hint>", and for multiple choice
#   "\nOptions: (A) .. (B) ..\nAnswer with the option's letter from the given choices directly." (Portuguese for
#   MMBench_dev_pt). None of VLMEvalKit's own instructions. Source: velvet-vision scripts/unroll_dataset_manager.py
#   + lib/dataset_manager/config/tasks/instruct/{context,mcqa}_suffix.json.
# - smolvlm (``build_prompt_for_dataset``): VLMEvalKit's prompt with SmolVLM-style rewrites.

MMBENCH = {
    'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN',
    'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11',
    'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11',
    'CCBench',
}
BRIEF = {'MME', 'MMVet', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'}
PURE_MCQ = {'MMStar', 'SEEDBench_IMG', 'AI2D_TEST', 'AI2D_MINI', 'ScienceQA_VAL', 'ScienceQA_TEST'}

# Image budget per benchmark, as the training evaluation of each tuning recipe uses it (velvet-vision
# configs/runs/2_tuning_slurm_<recipe>.yaml, vlmevalkit/* entries); multi-view exports only. Datasets not listed
# use the export's own defaults. The recipe comes from VELVET_VISION_RECIPE (default v9: every export up to v9).
VISION_OVERRIDES_BY_RECIPE = {
    'v9': {
        'DocVQA_VAL': {'max_tiles': 12},
        'InfoVQA_VAL': {'max_tiles': 12},
        'ChartQA_TEST': {'zoom': 2},
        'TextVQA_VAL': {'zoom': 1.5},
        'OCRBench': {'zoom': 1.5},
        'ScienceQA_VAL': {'zoom': 1.5},
    },
}
# v10 trains and evaluates infographics at 24 views and zoom 1.5 (VLMEvalKit's InternVL budget for InfoVQA).
VISION_OVERRIDES_BY_RECIPE['v10'] = {
    **VISION_OVERRIDES_BY_RECIPE['v9'],
    'InfoVQA_VAL': {'max_tiles': 24, 'zoom': 1.5},
}
DEFAULT_RECIPE = 'v9'


def vision_recipe():
    recipe = os.environ.get('VELVET_VISION_RECIPE', DEFAULT_RECIPE)
    if recipe not in VISION_OVERRIDES_BY_RECIPE:
        raise ValueError(f'VELVET_VISION_RECIPE={recipe!r}: known recipes are {sorted(VISION_OVERRIDES_BY_RECIPE)}')
    return recipe


VISION_OVERRIDES = VISION_OVERRIDES_BY_RECIPE[DEFAULT_RECIPE]  # v9's table, for code that reads it directly


# Answer budget per benchmark in the training evaluation (configs/runs/2_tuning_slurm_v*.yaml,
# vlmevalkit/* entries: max_new_tokens), the same for every tuning recipe.
TRAINING_MAX_NEW_TOKENS = {
    'AI2D_TEST': 16, 'COCO_VAL': 64, 'CVQA_EN': 16, 'ChartQA_TEST': 32, 'DocVQA_VAL': 32, 'InfoVQA_VAL': 32,
    'MMBench_DEV_EN': 16, 'MMBench_dev_pt': 16, 'MME': 8, 'MMMU_DEV_VAL': 16, 'MMStar': 16, 'OCRBench': 64,
    'POPE': 8, 'RealWorldQA': 16, 'ScienceQA_VAL': 16, 'TextVQA_VAL': 32,
    # Not in the training suite: MMBench's dev + test (answers in the internal MMBench.tsv) and its test split,
    # prompted like MMBench_DEV_EN.
    'MMBench': 16, 'MMBench_TEST_EN': 16,
}
TRAINING_MCQ = {
    'AI2D_TEST', 'CVQA_EN', 'MMBench_DEV_EN', 'MMBench_dev_pt', 'MMMU_DEV_VAL', 'MMStar', 'RealWorldQA',
    'ScienceQA_VAL', 'MMBench', 'MMBench_TEST_EN',
}
TRAINING_LANGUAGE = {'MMBench_dev_pt': 'pt'}
# Only MMMU places its images at the <image k> references (VLMEvalKit's split_MMMU); elsewhere they stay text.
TRAINING_INTERLEAVED = {'MMMU_DEV_VAL'}
COCO_CAPTION_PROMPT = (
    'Please describe this image in general. Directly provide the description, '
    'do not include prefix like "This image depicts". '
)
CONTEXT_SUFFIX = {
    'en': '. You also know this: {context}',
    'pt': '. Também sabes isto: {context}',
}
MCQA_SUFFIX = {
    'en': "\nOptions: {options}\nAnswer with the option's letter from the given choices directly.",
    'pt': '\nOpções: {options}\nResponde diretamente com a letra da opção escolhida entre as indicadas.',
}
OPTION_LETTERS = 'ABCDEFGHI'
IMAGE_REFERENCE = re.compile(r'<image (\d+)>')


def _cell(line, key):
    """A TSV cell as the training converter reads it: None when missing, blank or "nan"."""
    value = line.get(key) if hasattr(line, 'get') else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text.strip() and text.strip().lower() != 'nan' else None


def restore_options(line, raw_options):
    """``line`` with the option cells VLMEvalKit's loader read as missing put back from ``raw_options``
    ({letter: raw TSV string}): pandas turns the literal options "NA" and "None" (AI2D_TEST, CVQA_EN) into
    NaN, while the training converter reads the TSV as text and shows them."""
    missing = [k for k in OPTION_LETTERS if k in raw_options and _cell(line, k) is None and _cell(raw_options, k)]
    if not missing:
        return line
    line = line.copy()
    for letter in missing:
        line[letter] = raw_options[letter]
    return line


def read_raw_options(tsv_path):
    """{index: {letter: cell}} of a dataset TSV read as text (see ``restore_options``). Streams the file:
    CVQA_EN's TSV is 6.7 GB of base64 images."""
    out = {}
    if osp.exists(tsv_path):
        csv.field_size_limit(sys.maxsize)
        with open(tsv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                out[row['index'].strip()] = {k: row[k] for k in OPTION_LETTERS if row.get(k)}
    return out


def has_missing_option(line):
    return any(k in line and line[k] != line[k] for k in OPTION_LETTERS)  # NaN != NaN


def uses_training_prompt(dataset):
    return dataset in TRAINING_MAX_NEW_TOKENS


def build_training_question(line, dataset):
    """The user text of the training evaluation for one TSV row (images not included)."""
    lang = TRAINING_LANGUAGE.get(dataset, 'en')
    question = _cell(line, 'question')
    if question is None and dataset == 'COCO_VAL':
        question = COCO_CAPTION_PROMPT
    question = question or ''
    hint = _cell(line, 'hint')
    if hint is not None:
        question += CONTEXT_SUFFIX[lang].format(context=hint)
    if dataset in TRAINING_MCQ:
        options = [(letter, _cell(line, letter)) for letter in OPTION_LETTERS]
        options = [(letter, text) for letter, text in options if text is not None]
        if options:
            question += MCQA_SUFFIX[lang].format(options=' '.join(f'({k}) {v}' for k, v in options))
    return question


def training_messages(question, images, dataset):
    """Interleave ``images`` (paths, the TSV's order) with ``question`` as the training evaluation does:
    MMMU puts image k at each <image k>; a row without references gets all its images first."""
    if dataset in TRAINING_INTERLEAVED and IMAGE_REFERENCE.search(question):
        # The chat template writes a newline between an image part and the text after it (consecutive images
        # stay side by side), where the training prompt has a bare "<image>": a newline that starts the following
        # text is dropped so the two agree there; elsewhere the template's newline stays (the only difference
        # left from the training prompt).
        msgs = []
        pos = 0
        for match in IMAGE_REFERENCE.finditer(question):
            k = int(match.group(1)) - 1
            if k >= len(images):
                raise ValueError(f'{dataset}: question references image {k + 1} of {len(images)}')
            text = question[pos:match.start()]
            if msgs and text.startswith('\n'):
                text = text[1:]
            if text:
                msgs.append(dict(type='text', value=text))
            msgs.append(dict(type='image', value=images[k]))
            pos = match.end()
        text = question[pos:]
        if text.startswith('\n'):
            text = text[1:]
        if text:
            msgs.append(dict(type='text', value=text))
        return msgs
    return [dict(type='image', value=p) for p in images] + [dict(type='text', value=question)]


def vision_overrides_for(dataset):
    """The image budget of ``dataset`` for the recipe in VELVET_VISION_RECIPE (None: the export's defaults)."""
    overrides = VISION_OVERRIDES_BY_RECIPE[vision_recipe()].get(dataset)
    return dict(overrides) if overrides else None


def build_prompt_for_dataset(prompt, dataset=None):
    """Dispatch to a dataset-specific prompt rewrite, mirroring SmolVLM's per-dataset prompting."""
    if dataset in MMBENCH:
        return build_prompt_mmbench(prompt)
    elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
        return build_prompt_mmmu(prompt)
    elif dataset in ['MathVista_MINI']:
        return build_prompt_mathvista(prompt)
    elif dataset in ['ChartQA_TEST']:
        return build_prompt_chartqa(prompt)
    elif dataset in ['DocVQA_VAL', 'DocVQA_TEST']:
        return build_prompt_docvqa(prompt)
    elif dataset in ['TextVQA_VAL', 'TextVQA_TEST', 'TextVQA_VAL_local']:
        return build_prompt_textvqa(prompt)
    elif dataset in BRIEF:
        return build_prompt_default(prompt, add_brief=True)
    elif dataset == 'HallusionBench':
        return build_prompt_default(prompt, add_yes_or_no=True)
    elif dataset in PURE_MCQ:
        return build_prompt_puremcq(prompt)
    else:
        return build_prompt_default(prompt)


def build_prompt_default(prompt, add_brief=False, add_yes_or_no=False):
    if add_brief:
        prompt += '\nGive a very brief answer.'
    if add_yes_or_no:
        prompt += '\nAnswer yes or no.'
    return prompt


def build_prompt_puremcq(prompt):
    replace_mapping = {
        '\nOptions:': '\nChoices:',
        'Please select the correct answer from the options above.': 'Answer with the letter.',
    }
    for k, v in replace_mapping.items():
        prompt = prompt.replace(k, v)
    return prompt


def build_prompt_mmbench(prompt):
    replace_mapping = {
        'Please select the correct answer from the options above.':
            'Answer with the option\'s letter from the given choices directly.',
    }
    for k, v in replace_mapping.items():
        prompt = prompt.replace(k, v)
    # Swap hint and question, same as SmolVLM
    if prompt.startswith('Hint:'):
        hint, question = prompt.split('\nQuestion:')
        question, choices = question.split('\nOptions:')
        hint = hint.replace('Hint: ', '"You also know this:')
        prompt = 'Question:' + question + '\n' + hint + '\nOptions:' + choices
    return prompt


def build_prompt_mmmu(prompt):
    replace_mapping = {
        'Please select the correct answer from the options above.': 'Answer with the letter.',
        '\nOptions:': '\nChoices:',
    }
    for k, v in replace_mapping.items():
        prompt = prompt.replace(k, v)
    if 'A.' in prompt and 'B.' in prompt:
        prompt += '\nAnswer with the letter.'
    return prompt


def build_prompt_mathvista(prompt):
    replace_mapping = {
        '(A) ': 'A. ', '(B) ': 'B. ', '(C) ': 'C. ', '(D) ': 'D. ',
        '(E) ': 'E. ', '(F) ': 'F. ', '(G) ': 'G. ', '(H) ': 'H. ',
        '\nOptions:': '\nChoices:',
        'Hint: ': '',
    }
    for k, v in replace_mapping.items():
        prompt = prompt.replace(k, v)
    if 'A.' in prompt and 'B.' in prompt:
        prompt += '\nAnswer with the letter.'
    return prompt


def build_prompt_chartqa(prompt):
    return prompt


def build_prompt_docvqa(prompt):
    return prompt


def build_prompt_textvqa(prompt):
    return prompt


class VelvetVLM(BaseModel):
    """Wrapper for the `velvet_vlm` connector checkpoints (SigLIP2 vision tower +
    MLP projector + Mistral language model), loaded via `trust_remote_code=True`
    from a local HF-export directory (see `modeling_velvet_vlm.py` in the checkpoint).
    """

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='AlmawaveLabs/Velvet2-3B-Vision', **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16
        ).eval()

        self.model = self.model.to("cuda")

        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    # Datasets of the training evaluation get its prompts (build_training_question), with every
    # image of the row in the prompt's order; the others keep VLMEvalKit's prompt with the SmolVLM-style rewrites.
    def use_custom_prompt(self, dataset):
        return uses_training_prompt(dataset) and self._is_multiview()

    def build_prompt(self, line, dataset=None):
        images = self.dump_image(line, dataset)
        images = images if isinstance(images, list) else [images]
        if dataset in TRAINING_MCQ and has_missing_option(line):
            if not hasattr(self, '_raw_options'):
                self._raw_options = {}
            if dataset not in self._raw_options:
                from ..smp import LMUDataRoot
                self._raw_options[dataset] = read_raw_options(
                    os.path.join(LMUDataRoot(), f'{dataset}.tsv'))
            line = restore_options(line, self._raw_options[dataset].get(str(line['index']), {}))
        question = build_training_question(line, dataset)
        return training_messages(question, images, dataset)

    def _is_multiview(self):
        return getattr(self.processor, '_is_multiview', lambda: False)()

    def generate_inner(self, message, dataset=None):
        if self.use_custom_prompt(dataset):
            # Text and images in the message's order, every image: the chat template writes "<image>\n" per image.
            content = [{'type': 'image'} if m['type'] == 'image' else {'type': 'text', 'text': m['value']}
                       for m in message]
            images = [Image.open(m['value']).convert('RGB') for m in message if m['type'] == 'image']
            chat = [{'role': 'user', 'content': content}]
            return self._generate(chat, images, dataset)
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        message_for_prompt = "You are Velvet-2-3B, a Large Language Model (LLM) created by Almawave Labs, an Italian company headquartered in Rome. You are an helpful assistant. Your goal is to provide clear, accurate, and useful and polite answers to the user"
        prompt = self.build_prompt_for_dataset(prompt, dataset)
        prompt = message_for_prompt + "\n" + prompt
        image = Image.open(image_path).convert('RGB')

        # Multi-view checkpoints need an explicit image content part so the chat template emits
        # the literal "<image>" marker that processing_velvet_vlm.py._call_multiview() counts;
        # legacy (fixed placeholder) checkpoints keep the plain string content as before.
        if getattr(self.processor, '_is_multiview', lambda: False)():
            chat = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        else:
            chat = [{'role': 'user', 'content': prompt}]
        return self._generate(chat, [image], dataset)

    def _generate(self, chat, images, dataset):
        # bos_token is already inserted by the chat template, so tokenize without adding it again
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # text_inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        # vision_inputs = self.processor(images=[image], return_tensors='pt')

        # Multi-view exports take the benchmark's own image budget, as in the training evaluation.
        processor_kwargs = {}
        overrides = vision_overrides_for(dataset)
        if overrides and getattr(self.processor, '_is_multiview', lambda: False)():
            processor_kwargs['vision_processing_overrides'] = overrides

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors='pt',
            **processor_kwargs,
        )

        device = self.model.device
        generate_kwargs = dict(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
        )
        if getattr(self.processor, '_is_multiview', lambda: False)():
            # vision_layout='views': pixel_values must stay uint8 (casting to the model dtype
            # breaks modeling_velvet_vlm.py's uint8 check), and generate() needs the view/token
            # bookkeeping tensors too, or it thinks no images were given.
            generate_kwargs['pixel_values'] = inputs['pixel_values'].to(device)
            for key in ('view_grid_hw', 'num_views_per_image', 'num_tokens_per_image', 'images_per_sample'):
                if key in inputs:
                    generate_kwargs[key] = inputs[key].to(device)
        else:
            generate_kwargs['pixel_values'] = inputs['pixel_values'].to(device=device, dtype=self.model.dtype)
            # NaFlex SigLIP2 vision towers additionally require these two tensors
            if 'pixel_attention_mask' in inputs:
                generate_kwargs['pixel_attention_mask'] = inputs['pixel_attention_mask'].to(device)
            if 'spatial_shapes' in inputs:
                generate_kwargs['spatial_shapes'] = inputs['spatial_shapes'].to(device)

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs, **self.kwargs)

        # NOTE: this model's generate() builds inputs_embeds internally (no input_ids
        # given to the underlying LM), so the returned ids are the *new* tokens only,
        # not prompt+continuation like most HF models - do not slice off the prompt length.
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()

    def build_prompt_for_dataset(self, prompt, dataset=None):
        return build_prompt_for_dataset(prompt, dataset)
